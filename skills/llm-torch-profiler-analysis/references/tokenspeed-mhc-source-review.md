# TokenSpeed V4.1 mHC fusion: merged implementation and capture limits

Inspected on 2026-11-08. The complete PR diff, body, 17 issue comments,
28 reviews and 21 inline comments were read. Current callers and kernel
wrappers were checked at TokenSpeed `7b21034e72bb45d1edef3d3b5483b50b58c4c492`.
The upstream GPU results below were not reproduced for this skill update.

## PR #1767 — MegaMHC prefill and fused all-reduce

- Link: https://github.com/lightseekorg/tokenspeed/pull/1767
- State: merged 2026-10-24; head `884a15519e8ea38458c735e6484c420b0fcd5fb8`,
  base `486e857c5753e2300cf60d7e796c2cbefb90ed36`,
  merge `3b684074c0ef79bc8bc00b31cda8dc2850e4b71f`.
  A fresh API check matched both reviewed revisions.
- Coverage: all 3,496 diff lines, 27 files, +2697/-158. The final revision
  includes graph-safety restrictions absent from the early proposal and CI plans.
- Motivation: materializing each sublayer's HC post before its consumer loses
  opportunities to combine residual mixing, the carried pre-mix, normalization
  and the next coefficient computation. Moving a reduction across that boundary
  also changes stream dependencies, so admission must precede the producer.

### Handoff and reduction ownership

`V41HCPost(x, post, comb, reduce_group)` carries the unfinished post. A nonempty
`reduce_group` means the producer already admitted communication fusion and
therefore returned a partial sum. Otherwise `x` is reduced, including outputs
from an expert-owned communication path. The consumer supplies its own norm
weight/epsilon and HC parameters; no next-layer objects cross the boundary.

`supports_all_reduce_mhc_norm` inspects the anticipated output's shape, dtype,
device, layout and prepared workspace before the producer skips its normal
all-reduce. CED narrowing and graph padding affect that anticipated shape.
Admission reads metadata without launching tensor operations. Once admitted,
execution returns residual/norm outputs or raises; it cannot silently select a
late fallback after moving the reduction past the HC coefficient join.
The ordinary path keeps reduction inside the producer's stream fork.

MoE deferral additionally requires `moe_output_all_reduced(ctx)`, an allowed
next consumer and no expert-owned communication. Slot-ordered combines and
query-sharded forwards do not owe that plain all-reduce. A stage end or next
Engram layer prevents deferral; `finish()` raises if a reduction nevertheless
reaches it unfinished. Encoder/narrowing/decoder handoffs contain completed
residuals, and DSpark taps observe the completed stream after Engram. Draft
layers share the handoff and complete the last post before final normalization.
Empty ranks still participate in required expert communication, while a
cache-only narrowing step skips unused HC coefficient work.

### Dispatch and numerical contracts

| Path | Final/current contract | Trace implication |
|---|---|---|
| MNNVL all-reduce + post + carried pre + RMSNorm | Contiguous CUDA BF16 `[T,5120]`, four residual streams, FP32 coefficients, BF16 norm weight; 2/4/8/16 ranks; `0<T<=2048`; prepared workspace and selected one-shot/two-shot capacity must fit. FP32 Lamport, deterministic RSAG and batch-invariant collective settings decline admission. | Separate reduction/post/norm can be a valid platform, shape, topology or policy fallback. Four HC streams do not imply TP4. |
| Shifted MegaMHC | SM100-family, HC4, positive hidden width divisible by 1024, `0<T<=2**20`, positive Sinkhorn iterations, exact contiguous colocated BF16/FP32 operand layouts and available DeepGEMM API. Consumes an already reduced output. | Fuses post, carried-pre collapse, RMSNorm and next coefficients. The carried pre is distinct from the newly generated pre. |
| Local coefficient overlap | Attention forks only for decode with a compatible auxiliary stream. FFN can fork when coefficients were not already produced, but prefill/mixed breakable capture disables that fork. | A missing side stream or missing coefficient kernel can follow the selected phase/capture path or coefficient reuse. |

The runtime selects communication fusion for eligible decode forwards, including
multi-token speculative decode batches. Otherwise decode prefers local
coefficient overlap when a stream is available; eager prefill/mixed forwards
can try MegaMHC first. TokenSpeed's `DECODE` mode includes one or more tokens per
request; use actual rows and mode rather than equating it with batch size.

The collective reuses the group's armed MNNVL workspace. One-shot payload
capacity scales with `T * world_size`; two-shot uses
`2 * ceil(T / world_size) * world_size` rows. The native binding validates
capacity, rank, operand types/devices and pointer/flag types before launch.
It uses the existing strategy resolver and PDL setting, with BF16 collective
accumulation (`fp32_acc=false`).

The epilogue preserves materialization boundaries: FP32 `post*x`, ordered FP32
FMAs for residual streams, BF16 updated residual, ordered carried-pre collapse,
then another BF16 value before RMSNorm. Portable Triton and gfx950 Gluon posts
gain explicit FMAs to match this ordering. The changed cancellation test catches
the difference between 1.5625 and 1.5546875 at a BF16 rounding boundary, including
a partial hidden block. This does not establish bitwise equality for all GPU
paths: the distributed and MegaMHC tests use numerical tolerances.

### Graph replay changes the available path

The [author's barrier investigation](https://github.com/lightseekorg/tokenspeed/pull/1767#issuecomment-6248512684)
reports that DeepGEMM's split barriers used `%gridid` as an execution tag.
A graph node can reuse that tag across replays; a waiter can see a previous
terminal tag before the current initializer runs and consume stale partials.
The B200 probe reports failures on 353/400 loaded graph replays at 33 rows,
versus 0/400 eager calls under the same load. This is an upstream diagnostic
report, not a local reproduction or a general failure rate for serving graphs.

The final model diff therefore contains:

```diff
+    # DeepGEMM mega_mhc split barriers race under CUDA graph replay: a replay can
+    # read the previous replay's partials, so keep it out of captured segments.
+    if not prefer_overlap and not is_breakable_capture_active():
+        shifted = try_mhc_shifted_post_pre_norm(
```

The low-level API and adapter document that capture/replay is unsupported;
their eligibility check does **not** automatically reject ordinary CUDA graph
capture. The model's explicit guard covers breakable capture. A new direct
caller must honor the API contract instead of relying on shape eligibility or
warmup to make capture safe. The final MegaMHC refresh test runs eagerly on two
streams; its stale module docstring still says “graph input refresh.” Likewise,
the design paragraph about warming MegaMHC for graph capture and the layer
comment promising FFN overlap in every phase are not the final executable gates.

Keep this separate from the MNNVL fused collective: its changed test still
captures and replays with refreshed inputs, both native/public entry points,
PDL on/off, strategy crossover rows and a reused workspace. The MegaMHC
restriction is not evidence that every mHC fusion is excluded from graphs.

### Files and validation evidence

All 27 reviewed files:

| Directory | Files |
|---|---|
| `docs/design/` | `unified_path.md` |
| `python/tokenspeed/runtime/distributed/` | `comm_ops.py`, `comm_backend/auto.py`, `comm_backend/base.py`, `comm_backend/trtllm_allreduce.py` |
| `python/tokenspeed/runtime/models/` | `deepseek_v41.py`, `deepseek_v41_dspark.py` |
| `test/runtime/` | `distributed/test_comm_ops.py`, `test_deepseek_v41_dspark.py`, `test_deepseek_v41_model.py`, `test_prefill_graph.py` |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/` | `mhc.py` |
| `tokenspeed-kernel/python/` | `setup.py` |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/` | `communication/trtllm.py`, `residual/__init__.py`, `residual/deep_gemm.py`, `residual/triton.py` |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/` | `mega_mhc.py`, `trtllm.py`, `csrc/trtllm_mnnvl_mhc.cu` |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/comm/` | `trtllm_allreduce_fusion.cuh`, `trtllm_mnnvl_allreduce_fusion.cuh` |
| `tokenspeed-kernel/test/nvidia/` | `ops/test_mega_mhc.py`, `test_trtllm_mnnvl_gate.py`, `thirdparty/test_trtllm_mnnvl_comm.py` |
| `tokenspeed-kernel/test/` | `ops/test_mhc_post.py`, `test_kernel_api_selection.py` |

Behavioral tests cover reduction-before-join ordering, consumer norm selection,
admitted-fusion errors, empty-rank participation, CED narrowing, Engram/taps,
stage handoffs, numerical output and graph input refresh. Prefill warmup tests
also verify the execution stream and allocation-observer ordering. The final
diff adds those tests but does not modify `prefill_graph.py` itself; that
implementation was already in the final base. Export-list assertions alone
are not comparable numerical or stream-safety evidence.

The PR body reports four B300s, V4.1-Flash attention/MoE TP4, baseline
`bce6d20a`, one warmup wave and medians of three measured waves:

- 8192-in/1024-out decode cycles: 2.41–4.87% reduction at batches 1–32.
- Five-step/six-draft-token speculation: verify-cycle reductions of 1.95–5.25%.
  Accepted-token counts do not directly scale this cycle metric into throughput.
- Batch-one prefill TTFT: 4.45–7.08% reductions for the reported 2K–32K points;
  the 1K difference is explicitly inconclusive against run variation.

These body measurements do not identify a measured candidate SHA and must not
be presented as a fresh benchmark of the final graph restrictions. The
[H20 discussion](https://github.com/lightseekorg/tokenspeed/pull/1767#issuecomment-6240024403)
also reports neutral/slightly positive Marlin results, but Triton MoE regressions
in a stacked branch, with the persistent stage-1 kernel and shared-expert GEMM
both taking longer. Its exact responsible change was not isolated; this is
reason to inspect contention and exposed tails, not a universal fusion win.

Repeated bot “CI plan” comments are recommendations, explicitly not test runs.
The final-head AMD benchmark bot reports 579 cases within budget against the
recorded base; this refresh did not inspect the run's checkout logs, and the
report is not a V4.1 serving-accuracy certification. A review also identifies
the removal of V4.1's AMD Iris post-only fusion caller. The final/current V4.1
path admits the TRT-LLM norm fusion only; the surviving Iris backend method
does not establish a reachable V4.1 call. Do not claim AMD fusion preservation
from that method's existence or the aggregate benchmark report.

## Current source beyond the merged PR

Current attention deferral uses `head_tp_group`/`head_tp_size`, while MoE uses
its own TP×EP group. Query-shard empty-rank participation has also evolved.
Use the current row/group contract when attributing collectives.

The current MegaMHC join can additionally emit E4M3 codes and packed group-32
UE8M0 scales for the reference-FP8 `wq_a_wkv` consumer with projection scales.
Scale storage is int32 `[T,H/128]`, stride `(1, T padded to 4)` for TMA alignment.
The BF16 normalized input remains an output. This optional quantized join is a
later addition, not part of #1767's final diff, and retains the same capture
restriction. A missing separate input-quantization kernel can be expected here.

Current source:
[V4.1 producer/consumer gates](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/models/deepseek_v41.py),
[DSpark handoff](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/models/deepseek_v41_dspark.py),
[MoE sum ownership](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/distributed/comm_manager.py),
[collective admission/execution](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/trtllm.py),
[MegaMHC eligibility](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/deep_gemm.py),
[BF16/FP8 adapter](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/mega_mhc.py),
[one-shot/two-shot workspace](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/trtllm.py).
