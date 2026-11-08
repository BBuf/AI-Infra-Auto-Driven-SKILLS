# DSA chunk top-k: row mapping versus page-table gathering

Reviewed on 2026-11-09 (Asia/Shanghai). No GPU or model run was performed.

## PR #39522 — direct indexing in the legacy AOT transform

- Link/title: [[Kernel] Fuse chunk topk transform with row_to_page_table direct inde…](https://github.com/sgl-project/sglang/pull/39522).
- State: open, not merged. Head `c46e07830454026cb054847140c74bbac106611c`,
  API base `c9fbe5f6552f26292f3ec3382a06cc1931f071cc`, actual merge base
  `5dde6e8f02e0992525f914a52868b89c7d56f64a`.
- Coverage: all 267 lines of the six-file diff read, +123/-33; full body,
  the single commit message and the single issue comment read. Paginated
  API queries returned no formal reviews or inline comments. The comment
  requests review/CI; it contains no additional validation evidence.
- Motivation: the chunked PAGED path materialized
  `page_table_1[batch_idx_list]` before selecting. For a row map of length Q
  and table width W this copies Q × W entries. Supplying the map directly
  removes that intermediate gather and the prefill kernel's cumulative-length
  search. It does not eliminate the original page-size-one table or logits.

The new [CUDA kernel](https://github.com/sgl-project/sglang/blob/c46e07830454026cb054847140c74bbac106611c/python/sglang/kernels/aot/csrc/elementwise/topk.cu)
selects its source table row directly:

```diff
+  // Direct indexing: each Q row maps to a batch row via row_to_page_table.
+  // No shared-memory scan of cu_seqlens_q needed.
+  const auto src_page_entry = src_page_table + row_to_page_table[bid] * src_stride;
```

The rest reuses the old top-2048 selection: one 1,024-thread CTA per score
row, the short-row copy/`-1` pad, or `fast_topk_cuda_tl` followed by two mapped
stores per thread. `row_starts` offsets score reads; selected positions are
relative to that request and index the table without adding `row_starts`.
Short rows need no score ranking, so their copy path does not read the offset.
The selection algorithm, tie handling and bounded threshold-bin staging are
inherited. Its staging can truncate a crowded bin; this patch adds no exact
overflow fallback and should not be described as an exactness improvement.

## Caller shape, API and validation boundaries

At this head, the actual chunk caller in
[dsa_indexer.py](https://github.com/sgl-project/sglang/blob/c46e07830454026cb054847140c74bbac106611c/python/sglang/srt/layers/attention/dsa/dsa_indexer.py)
supplies `token_to_batch_idx[start:end]`: one request ID per query row.
It treats each row as a length-one query sequence. The
[metadata adapter](https://github.com/sgl-project/sglang/blob/c46e07830454026cb054847140c74bbac106611c/python/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py)
still computes cumulative lengths and a repeated offset before the backend
call. The new kernel avoids reading those cumulative lengths; this diff does
not remove their earlier construction.

The [backend change](https://github.com/sgl-project/sglang/blob/c46e07830454026cb054847140c74bbac106611c/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py)
applies only after fused-top-k admission, on the SGL_KERNEL PAGED fallback
when `batch_idx_list` is present. Earlier decode-shaped v2 and ragged-v2
branches remain ahead of it; other top-k backends do not use the new kernel.
It converts the mapping to contiguous int32 on the logits device and passes
the original table. Python-list input can therefore still allocate/copy a
mapping; “no intermediate work” is too broad.

The C++ mapped branch checks a contiguous one-dimensional int32 mapping of
exactly `score.size(0)` entries. It does not check its values against table
row bounds, validate table column coverage, or normalize negative IDs as
PyTorch advanced indexing does. Production-generated nonnegative request
IDs meet the intended contract. A list containing one ID per **request**,
with several query rows per request, must first be expanded; merely casting
the list produces the wrong length. The inspected historical chunk caller
already supplies per-row IDs, so that general API limitation is not evidence
of a failure in that caller.

The [Python wrapper](https://github.com/sgl-project/sglang/blob/c46e07830454026cb054847140c74bbac106611c/python/sglang/kernels/aot/python/sgl_kernel/top_k.py)
still requires K=2048 and now always sends a seventh argument, possibly
`None`. Both CUDA and ROCm schemas and the C++ declaration are changed.
The registered schema marks the argument nullable but does not give it a
default. Rebuild/install the matching AOT extension: Python's optional
parameter does not make a seven-argument call compatible with an older
six-argument binary, nor preserve old direct `torch.ops` arity automatically.
The nonmapped branch retains its cumulative-length decode/prefill dispatch.

## Current main already avoids some of this gather

At `c24285bde97c3e6424cd5d8cb35788889034d8ea`, the
[DSA backend](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py)
still contains the old AOT gather, but it is a fallback. The proposed
`row_to_page_table` AOT parameter is not present. Determine the actual branch
before attributing a trace to this outstanding optimization:

- With SGL_KERNEL and `SGLANG_OPT_USE_TOPK_V2`, eligible packed PAGED extend
  already passes `token_to_batch_idx` or the chunk's tensor row map to
  `topk_transform_packed_v2`. Admission requires FP32, unit column stride,
  row stride divisible by four, row starts, matching row counts and
  `0 < K <= 2048`. Prefill-CP shards additionally need `packed_extend_topk`.
  Request-level Python lists stay on the legacy route.
- The current [packed-v2 implementation](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh)
  indexes a compact page table through `row_to_batch`, then transforms the
  row-local selected position using the actual page size. It needs no
  per-forward top-k plan. It can mask up to three columns before an unaligned
  score window **in place**; score rows must not overlap and the buffer must
  have no later consumer. Nonnegative lengths and valid row-map values remain
  required. This differs from the proposed read-only FP32 AOT score input.
- Decode-shaped v2 consumes the compact table directly and may omit
  `page_table_1` entirely. A fallback to the old wide-table kernel is not
  always available on that path.
- The FlashInfer backend already passes `row_to_batch` to its page-table
  transform. Its adapter expands request-level mappings with cumulative
  query lengths when needed, using a known `output_size` to avoid a host
  synchronization for the repeat count. This is a separate existing route,
  not proof that the SGL_KERNEL AOT fallback was updated.

This reconciliation inspects current dispatch, wrappers, host checks and
packed addressing/masking. It does not count the entire v2 selection-kernel
history or FlashInfer implementation as a newly completed PR review.

## Tests, CI and performance evidence

No test or benchmark file is added by the six-file patch. Useful validation
would compare the mapped path with the original gather for repeated and
nonidentity request IDs, nonzero score offsets, short/empty/long rows,
threshold ties, strided tables, CUDA graphs and matching CUDA/ROCm binaries.
The optional mapping must also leave nonmapped dispatch working. These are
coverage requirements, not tests performed by this review.

The body reports GLM-5.2 scores of 0.8367 on tau2, 0.9531 on tau2 telecom,
and 0.9287 on GPQA. It supplies no matched baseline scores, evaluation
configuration, sample counts or launch recipe; those numbers do not prove
unchanged model accuracy. The claimed gather cost of up to 10 ms is also
not accompanied by a reproducible before/after benchmark or service result.

Exact-head paginated queries returned 15 workflows, 119 checks and no
commit statuses. Base [34928438580](https://github.com/sgl-project/sglang/actions/runs/34928438580)
and extra [34928438416](https://github.com/sgl-project/sglang/actions/runs/34928438416)
fail because the CI maintenance gate was active; their logs confirm a full
pause. AMD [34928438601](https://github.com/sgl-project/sglang/actions/runs/34928438601)
instead fails the missing-`run-ci` label gate. Relevant build/kernel/model
jobs are skipped. Successful lint/router checks do not validate this mapped
kernel, and infrastructure gates are not numerical test failures.

All six changed files were reviewed: `topk.cu`, CUDA/ROCm registration,
`sgl_kernel_ops.h`, `top_k.py` and `dsa_topk_backend.py`. The practical profiling
question is whether the observed gather belongs to the remaining legacy
fallback, and whether its removal saves the measured chain after current
dispatch and metadata costs are included.
