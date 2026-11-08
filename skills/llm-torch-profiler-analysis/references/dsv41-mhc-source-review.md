# DeepSeek V4.1 mHC state and collective ownership

Inspected on 2026-11-08. The complete PR diff, body, 24 issue comments and one
review below were read; there are no inline comments. Current dispatch and
boundary callers were checked at SGLang `c24285bde97c3e6424cd5d8cb35788889034d8ea`.
The [current mHC matrix](fuse-overlap-catalog.md#residual-mixing-and-collective-epilogues)
records the wider present-day kernel gates. Upstream test/performance reports
were not reproduced locally for this documentation update.

## PR #42245 — Unify mHC state and remove combine-only variants

- Link: https://github.com/sgl-project/sglang/pull/42245
- State: merged 2026-10-06; head `cf7c85937ba242f886202797fc654b948a723800`,
  base `150f568900db878ed3cf179ae896805d78c55747`,
  merge `730f1f3e5be9c781f523f856fc39c18c2ad266d6`.
  A fresh API check matched both recorded revisions.
- Coverage: full 3,812-line diff, 19 files, +1363/-1507. This is the final
  combined change, including the later collective widening and naming pass;
  the initial body and earlier review results do not describe every final detail.
- Motivation: previous-sublayer pre coefficients, residual streams, pending HIP
  post operations and precomputed inputs crossed layers through several parallel
  variables and context-local hooks. That obscured which component owned a
  collective and made cached inputs unsafe when residual rows changed.

`HcState(streams, pre, input)` represents the handoff. `streams` is a materialized
residual or `HcPending`; `input` is `HcNormed` or `HcQuantized`. A rewritten
residual invalidates its precomputed input. The original `take_rows` narrows
the residual/pending operands and pre coefficients while dropping the shortcut.
`release()` clears the consumed state so older Python frames stop retaining dead
intermediates; the caller separately retains the residual needed by post.

`HcSubLayer` holds parameter references without registering new checkpoint keys.
`HcConfig` carries the model/shape policy, while `combine`, `mix_stats` and `post`
own kernel dispatch. Weight refresh always rebinds hyper-connections and
invalidates cached fusion gates/boundaries, even when DeepGEMM lacks the optional
prenorm API. Hopper's compensated BF16 path remains independent of that API.

Attention explicitly returns `AttnOutput(partial)` only when its projection
actually defers the TP sum. MoE can return `MoEOutput` with routed/shared pieces
and a deferred-finalize handle. The boundary inspects those types; merely asking
to defer does not prove the producer complied. CP, gather/scatter, replicated
shared experts and reductions already owned elsewhere constrain the handoff.
The MXFP4 TRT-LLM dual-stream path attempts deferred finalize for
`return_moe_output` even though it was outside the older generic whitelist.

When a fused collective cannot consume a `MoEOutput`, fallback merges sharded
pieces, reduces once and then adds a replicated shared expert according to rank
policy. Putting that replicated term inside the sum would count it multiple times.

Exact fallback excerpt:

```diff
+        pieces = out
+        out = post_experts_all_reduce(pieces.get_merged())
+        if (
+            pieces.shared is not None
+            and pieces.shared_is_replicated
+            and should_add_replicated_moe_output()
+        ):
+            out += pieces.shared
```

The PR deletes `all_reduce_mhc_combine.py`, `mhc_post_combine.py` and
`mhc_post_fusion.py`, including the medium-batch combine-only variants and
`MhcPostFusion` context hook. Surviving collective entry points become
`all_reduce_mhc_post_combine_norm` and
`moe_finalize_all_reduce_mhc_post{,_combine_norm,_combine_norm_quant}`.
Eligible Blackwell combine/norm no longer has a medium-row hole; this is not an
unconditional fusion on every platform, norm configuration or dtype.

The final revision registers the TP group's existing `CustomAllReduceV2` push
plane before capture instead of creating a dedicated 4 MiB plane. Eligibility
checks registered world size, at most 384 true token rows, available phase
counters and slot bytes. Expanded expert-GEMM rows are not the token count.
The quantized collective epilogue grows from eight to **128** rows because its
scale writer addresses one 128-row tile; more rows need a different swizzle.

The stats stream forks immediately before combine, predicts coefficients beside
the sublayer and joins before post reads them. Wide-prefill stats-stream overlap
is removed by this PR. Hopper retains the small decode policy: one row, or
decode rows 2–64; target verify with more than one row does not use this stream.
AMD keeps its separately gated fused boundary and explicit `MhcPostOperands`;
requested fusion is checked with assertions instead of silently falling back.
The XPU split/Sinkhorn dispatch and Ascend batched-post form are preserved.

All reviewed files, grouped by directory:

| Directory | Files |
|---|---|
| `python/sglang/kernels/jit/csrc/distributed/` | `all_reduce_fusion.cuh` |
| `python/sglang/kernels/ops/communication/` | `all_reduce_mhc.py`, deleted `all_reduce_mhc_combine.py` |
| `python/sglang/kernels/ops/layernorm/` | `hc_combine_norm.py`, deleted `mhc_post_combine.py` |
| `python/sglang/srt/layers/` | deleted `moe/mhc_post_fusion.py`, `moe/moe_runner/flashinfer_trtllm.py`, `quantization/mxfp4_flashinfer_trtllm_moe.py` |
| `python/sglang/srt/models/deepseek_common/amd/` | `deepseek_v2_hip_moe.py`, `deepseek_v4_fused_mhc.py`, `deepseek_v4_hip.py` |
| `python/sglang/srt/models/` | `deepseek_v2.py`, `deepseek_v4.py`, `deepseek_v4_dspark.py`, new `deepseek_v4_mhc.py` |
| `test/registered/kernels/ops/layernorm/` | `test_mhc_aiter_hip.py`, `test_mhc_kernels.py`, `test_mxfp8_epilogue.py` |
| `test/registered/unit/models/` | `test_deepseek_v4_amd_tp4.py` |

Tests migrate existing Hopper numerical and changed-input graph coverage onto
the new API, cover missing-DeepGEMM cache refresh, and exercise four sharded/
replicated/absent shared-output cases with exactly one reduction. MXFP8 tests
extend the boundary sweep through 9, 17, 33, 96, 127 and 128 rows. AMD TP4 tests
explicitly decline fusion when another path owns the reduction. These behavioral
tests are worth retaining; they are not redundant API-name assertions.

The first author comment reports GSM8K 200-question scores of 0.885 for TP4 and
0.875 for TP4/EP4 on four GB300s, matching its then-current reference. It predates
the later revisions. The discussion records H100 and CPU reruns after successive
conflict repairs, including 50 H100 tests and 13 CPU tests at `bf26114627bd`.
A diagnostic AMD job actually ran the TP4 file with 12 passes on that earlier
head, while its enclosing workflow still failed on other files. That is
file-level evidence, not a green AMD gate for the final revision.

The [last rerun bot report](https://github.com/sgl-project/sglang/pull/42245#issuecomment-5999997405)
lists successful H100 mHC, B200 MXFP8, CPU boundary and H200 collective jobs.
It explicitly rejects the AMD-only files because that rerun workflow has no AMD
runner. The refresh did not inspect those job checkout logs or rerun them, and
does not turn the earlier accuracy result into final-head performance/accuracy
certification. No quantitative final-head end-to-end speedup is supplied here.

## What changed again before the frozen main

Current `deepseek_v4_mhc.py` still owns V4.1 kernel dispatch, but
`deepseek_v4_boundary.py` now connects reads, pending updates and sum ownership
to the general boundary system. The similarly named
`layers/layer_boundary/residual/mhc.py` is a different residual abstraction;
do not substitute it for V4.1's previous-pre dispatch source.

- Current fused-collective specializations support TP2/TP4, with caller and
  communicator-capacity checks. The registration helper now also accepts an
  optional dedicated push plane; the inspected DeepSeek V4 caller uses its
  no-argument form before capture. The old fixed 4 MiB assumption is obsolete.
- `HcState` now carries cached next statistics. `take_rows` can preserve narrowed
  normalized input/statistics, but drops quantized input because its scale layout
  is tiled. `with_residual` drops both cached input and statistics. Apply the
  current rule when interpreting late-layer reuse, not the original PR's broader
  shortcut invalidation.
- Decode post/combine/norm and opt-in Mega mHC prefill add new fusions beyond
  this PR. The old combine-only files remain deleted. Their absence does not
  establish a regression, and their historical row windows do not bound the
  newer kernels. Mega mHC can publish next coefficients, eliminating a later
  stats launch without adding an independent side-stream computation.
- `AttentionTransport` and `MoeTransport` declare and preserve row/sum ownership.
  CP, attention scatter and MoE gather can decline fused handoff. Replicated
  shared outputs still belong after reduction, exactly once. Inspect the
  returned `UnreducedOutput`/`MoeFinalize` and completion path, not only a flag.

Current source:
[V4.1 mHC state/dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_mhc.py),
[boundary ownership/transport](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_boundary.py),
[model gates and cache refresh](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4.py),
[MoE output handoff](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v2.py),
[communicator capacity](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py),
[collective wrappers](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/communication/all_reduce_mhc.py).
