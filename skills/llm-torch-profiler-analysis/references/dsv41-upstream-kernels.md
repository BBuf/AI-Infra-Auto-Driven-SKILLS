# DSV4.1: DeepGEMM, FlashMLA and attention top-k integration

Read this when attributing DSV4.1 indexer or sparse-attention kernels, changing
KV storage, or deciding whether a DeepSelect optimization applies. These five
PRs were authored by Ziyi Xu (`DarkSharpness`). Their full diffs, including tests,
were read on **2026-09-22**. Current dispatch was checked separately at SGLang
main [`771c9d782d9e`](https://github.com/sgl-project/sglang/commit/771c9d782d9ecf0324e70b7f5a08c32644d652c5).
This is implementation evidence, not a new GPU performance or accuracy result.


**2026-10-05 source refresh:** DSV4.1 is served from SGLang main
[`b1bbd74f287f`](https://github.com/sgl-project/sglang/commit/b1bbd74f287f13ed1276b0403a01ebb55c597e93);
the historical `dsv4.1` branch has been removed. Historical PR merge bases and
experiment pins below retain their original dates. Main still sets
`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=False` and `SGLANG_DSV4_KV_LAYOUT=v4`.
The WO-A caller still checks V4.1, BF16 weights, `(2,1024)` local group/rank
shape and the token bound; `SGLANG_DSV41_FUSED_WO_A=True` is not dispatch proof.
See the refreshed [mHC matrix](dsv41-mhc-fusions.md) and
[source contracts](../../../docs/upstream-source-contracts.md). No GPU rerun
or claim of numerical equivalence across these revisions is included.

## Choose the right operation

| Operation | Integration | Boundary to preserve |
|---|---|---|
| Dense source-indexer scores → candidate blocks → sparse consumer scores | #38944, DeepGEMM paged sparse MQA logits | Block selection, score GEMM and final token selection are separate stages. |
| Long-context FP32 score selection | #38829, DSA top-k v2 | Register, streaming and cluster paths have different eligibility. |
| Selected logical token IDs plus physical cache IDs | #39098, dual-output top-k v2 | Produce both from the same selection and preserve `-1` padding. |
| Short BF16 consumer-score rows → physical cache IDs | #39305, adapted DeepSelect | Attention token top-k, not MoE expert routing. |
| Quantized attention KV storage consumed by FlashMLA | #39123, V4.1 FP8/FP4 layouts | Writers, page strides, reader and prefill dequantization must agree. |

The 128-dimensional index-K FP4 pool uses **68 B/token** (64 payload + 4 scale
bytes). It is distinct from the 512-dimensional attention KV formats below.
Do not transfer their memory arithmetic or quantization groups to each other.

## Current dispatch and KV storage contract

At the pinned main revision, `make_candidate_indexer` returns no DeepGEMM
candidate object on pre-SM100 devices or when candidate-block selection is
disabled by the model configuration. Otherwise it requires
`DEEPGEMM_PAGED_SPARSE_MQA_LOGITS`; missing support raises an error naming
`sgl-deep-gemm >= 0.2.0` and `SGLANG_ENABLE_JIT_DEEPGEMM`. The original PR's
`SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER` opt-in is no longer the dispatch switch.
Hopper and prefill have separate paths; follow the caller instead of forcing
the Blackwell decode recipe onto them. See the
[current factory](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py)
and [DeepGEMM implementation](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py).

| Attention KV layout | Payload per stored token | Scales per token | Total before page padding | Page alignment |
|---|---|---|---|---|
| `V4` | 448 FP8 no-RoPE values + 64 BF16 RoPE values = 576 B | 7 UE8M0 scales + 1 padding byte = 8 B | 584 B | 576 B |
| `V41` | All 512 values, including RoPE, in FP8 E4M3 = 512 B | 16 UE8M0 scales, one per 32 values = 16 B | 528 B | 512 B |
| `V41_FP4` | All 512 E2M1 values, two per byte = 256 B | 32 E4M3 scales, one per 16 values = 32 B | 288 B | 256 B |

A page stores **all payload rows, then all scale rows**, followed by alignment
padding. It is not an array of interleaved 528-byte or 288-byte records:

```text
scale_region_offset = page_size * data_bytes
page_bytes = ceil(page_size * bytes_per_token / page_align) * page_align
```

For a two-token page, V41 allocates 1536 B and V41_FP4 allocates 768 B. Capacity
must additionally account for compression ratios, source-layer sharing,
index-K pools and compression state. `tokens * 528` is not whole-model KV use.
The [layout definition](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/ops/attention/dsv4/kv_layout.py)
and [pool selection](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
are the authority for addressing and allocation.

Current defaults remain `SGLANG_DSV4_KV_LAYOUT=v4` and
`SGLANG_DSV4_COMPRESSED_KV_LAYOUT=auto`. Selecting `v41` needs an SM100/SM103
GPU and a compatible installed FlashMLA reader. `auto` chooses V41 only when
the device and reader support it; the current capability probe checks the
reader's advertised 528-byte format. Merely having the SGLang source does not
upgrade the installed reader. Under V41, compressed `auto` uses FP4 for ratios
1/2, whose latents already have the matching FP4 rounding contract, and FP8 for
ratios 4/128. Forcing FP4 for all ratios changes that quantization contract.
The main SWA cache remains FP8. HiSparse's V4-only transfer path and the
separate unified-KV implementation must not inherit this format implicitly.

Trace the complete producer/consumer chain when changing a format:

1. Norm and RoPE preserve the required BF16 rounding boundaries before packing.
2. SWA writers and each compression-ratio writer select the same layout as its pool.
3. The pool allocates padded page strides and passes the correct views to decode.
4. FlashMLA reads that format; prefill gathers/dequantizes the same bytes into
   its BF16 workspace. HiCache transfers must respect the supported page layout.

## Diff review coverage

All five were merged into `dsv4.1` originally; current behavior above was
checked in main separately. Counts below are the complete fetched diffs, not
selected hunks. Every listed line was read; no truncation or deferred file.

| PR | Merge time (UTC) | Add / delete | Files | Diff lines read / fetched |
|---|---|---|---|---|
| #38829 | 2026-09-10 11:41:50 | +808 / -509 | 7 | 2050 / 2050 |
| #38944 | 2026-09-11 18:24:18 | +2520 / -207 | 19 | 3094 / 3094 |
| #39098 | 2026-09-11 14:21:51 | +165 / -16 | 5 | 390 / 390 |
| #39123 | 2026-09-13 16:50:41 | +2313 / -247 | 27 | 3792 / 3792 |
| #39305 | 2026-09-13 15:13:40 | +445 / -228 | 3 | 831 / 831 |

### #38829: DSA top-k v2 cluster rework and NaN padding

[PR: DSA top-k v2: long-context cluster rework, and NaN padding for the lanes outside a row](https://github.com/sgl-project/sglang/pull/38829)
— merged. [Reviewed head](https://github.com/sgl-project/sglang/commit/e830cebe9ca4829bd5270546790e82bbd0dfcbcd).

**Motivation.** Long rows need a different selection strategy from rows that
fit in registers. Padding with negative infinity also lets invalid lanes tie
with legitimate negative-infinity scores, corrupting selected indices.

**Implementation.** The planner uses register-resident variants for rows up to
8192/16384 scores, then streaming or clustered selection based on shape and
architecture. Cluster variants aggregate local histograms through distributed
shared memory; rank zero resolves the pivot and publishes it with cluster
synchronization. A driver occupancy probe supplies a cluster scheduling bound,
avoiding the assumption that SM count divided by cluster size is sufficient.
It is a scheduling probe, not a measurement of the real kernel's occupancy.

Cluster support is gated to CUDA architectures from SM90 through below SM110;
unsupported architectures retain the streaming path. Invalid lanes use NaN,
whose ordered comparisons do not enter selection. The diff also fixes
round-to-nearest-even bin boundaries and separates page transformation from
the selection problem. The padding contract is explicit in
[`topk_impl.cuh`](https://github.com/sgl-project/sglang/blob/e830cebe9ca4829bd5270546790e82bbd0dfcbcd/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh):

```cpp
constexpr float padding_value() {
  return std::numeric_limits<float>::quiet_NaN();
}
```

**Reviewed files.** CUDA implementation: `jit/csrc/deepseek_v4/topk_v2.cuh`,
`jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`, `jit/csrc/misc/probe.cuh`.
Python dispatch: `ops/attention/dsv4/topk.py`, `ops/misc.py` (all under
`python/sglang/kernels/`). Also reviewed
`test/registered/kernels/benchmark/attention/bench_topk.py` and
`test/registered/kernels/ops/attention/test_topk_v2.py`.

**Validation implications.** Exercise both register cutoffs and long clustered
rows, ragged starts, valid `-inf` ties, page transforms and graph replay. The
ragged path can write NaN into up to three alignment-prefix score entries after
its dependency wait: another consumer must not require the overwritten scores.
Do not infer cluster availability or correctness from a single dense-row case.

### #38944: two-level indexer on DeepGEMM paged sparse MQA logits

[PR: Enable the two-level candidate indexer on DeepGEMM's paged sparse MQA logits](https://github.com/sgl-project/sglang/pull/38944)
— merged. [Reviewed head](https://github.com/sgl-project/sglang/commit/6862d39295d39bdb0f5735e41e7e56ba29ab1d32).

**Motivation.** Consumer indexer layers need scores only for blocks nominated
by an index-source layer. Computing full-context logits and masking afterward
wastes work and memory traffic.

**Implementation.** The source still computes dense logits and its own token
selection. A second branch takes the maximum over each eight-token block,
forces the newest block to remain eligible with `+inf`, selects 2048 blocks,
then sorts block IDs using a bitmap/counting pass. It publishes logical block
IDs, physical block tables, lengths and DeepGEMM schedule metadata for reuse
by later consumer layers within the forward. Unused physical block slots carry
an `INT32_MAX` sentinel.

Consumers compute BF16 scores only for the candidate blocks: 2048 × 8 = 16384
candidate positions before final token top-k and page transformation. Consecutive
verify rows of one request can share a KV pass, while row-specific causal
lengths remain enforced. The central call in the original
[`candidate_deep_gemm.py`](https://github.com/sgl-project/sglang/blob/6862d39295d39bdb0f5735e41e7e56ba29ab1d32/python/sglang/srt/layers/attention/dsv4/candidate_deep_gemm.py)
is:

```python
return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
    (q_fp4, q_sf),
    k_cache,
    weights,
    table.schedule,
    table.blocks.shape[1],
    CANDIDATE_BLOCK_SIZE,
)
```

Candidate preprocessing can overlap the source's own top-k on a side stream;
the first consumer waits, and `record_stream` protects input storage lifetime.
Short sequences that fit wholly inside the candidate budget use dedicated
capture variants and can bypass selection. Index-K pages use 128 slots of
68 B/token, satisfying DeepGEMM's 512-byte page-stride alignment.

**Reviewed files.** Under `python/sglang/`: kernels `amax_copy.cuh`,
`sort_idx.cuh`, `topk_bf16_small.cuh`, `topk_v2.cuh`; wrappers
`ops/attention/dsv4/{candidate_blocks,topk,attn}.py`; runtime
`layers/attention/dsv4/{candidate_indexer,candidate_deep_gemm,candidate_torch,indexer}.py`,
`layers/attention/deepseek_v4_backend.py`, `srt/environ.py`, and
`mem_cache/deepseek_v4_memory_pool.py` (runtime paths under `srt/`). Tests:
`attention/unittests/dsv4/test_dsv41_sparse_indexer.py`,
`kernels/ops/attention/test_{amax_copy,sort_idx,topk_bf16}.py`, and
`kernels/test_dsv4_indexer_postprocess.py`, all under `test/registered/`.

**Validation implications.** Compare sparse scores with a gather from dense
reference scores using identical quantized inputs; include newest-block
retention, permuted physical pages, variable lengths and paired verify rows.
Check publish/wait ordering and capture reuse. The PR's original BF16 top-k is
superseded by #39305; its initial opt-in flag is not current-main guidance.

### #39098: raw-index output without falling back from top-k v2

[PR: Support raw-index output in TopK v2 (port of #33672)](https://github.com/sgl-project/sglang/pull/39098)
— merged. [Reviewed head](https://github.com/sgl-project/sglang/commit/d4bbeb68a34c7c32f4b7b7280b8f077d14ea0e12).

**Motivation.** Some verify and nonpaged-prefill consumers need logical token
indices as well as physical KV locations. Previously requesting raw indices
excluded the v2 fast path.

**Implementation.** A compile-time `DUAL_OUTPUT` mode writes both views of the
same selected set, preserving `-1` for invalid entries. The indexer now permits
v2 when raw output is requested and prefers the persistent
`c4_sparse_raw_indices` buffer over capture scratch storage. The trivial
short-row transform illustrates the contract in
[`topk_v2.cuh`](https://github.com/sgl-project/sglang/blob/d4bbeb68a34c7c32f4b7b7280b8f077d14ea0e12/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh):

```cpp
problem.out[tx] = tx < problem.seq_len ? transform.page_to_indices(tx) : -1;
if constexpr (kMode == TopKMode::DUAL_OUTPUT) {
  transform.raw_out[tx] = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
}
```

**Reviewed files.** `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`,
`python/sglang/kernels/ops/attention/dsv4/topk.py`,
`python/sglang/srt/layers/attention/dsv4/indexer.py`,
`test/registered/kernels/ops/attention/test_topk_v2.py`, and
`test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`.

**Validation implications.** Invert page mapping and compare with raw indices;
check ties, padding and the actual persistent output used during capture.
Retain the XPU fallback. This removes a dispatch restriction, not the score
GEMM or a communication operation.

### #39123: FlashMLA V4.1 FP8/FP4 paged KV layouts

[PR: Paged KV cache layouts for FlashMLA's V4.1 fp8 / fp4 formats](https://github.com/sgl-project/sglang/pull/39123)
— merged. [Reviewed head](https://github.com/sgl-project/sglang/commit/d0b30a5f3c3409649c0aae825320c7c1b2c612fb).

**Motivation.** A newer FlashMLA reader cannot consume the old 584-byte layout
as though it were the 528-byte FP8 or 288-byte FP4 format. Switching only the
attention call leaves incompatible writer offsets, page strides and scales.

**Implementation.** Shared Python/C++ layout traits define payload bytes,
scale groups and padded page strides. Every writer passes a layout template;
`PagedKV` computes data and scale offsets consistently. For example,
[`KVLayoutTraits<V41>`](https://github.com/sgl-project/sglang/blob/d0b30a5f3c3409649c0aae825320c7c1b2c612fb/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh)
defines:

```cpp
static constexpr int64_t kDataBytes = 512;
static constexpr int64_t kScaleBytes = 16;
static constexpr int64_t kTileSize = 32;
static constexpr int64_t kPageAlign = 512;
static constexpr int64_t kBytesPerToken = kDataBytes + kScaleBytes;
```

Norm/RoPE writers preserve BF16 rounding before quantization. FP4 uses rounded
E4M3 scales and IEEE division; replacing it with approximate reciprocal
multiplication can change rounding ties. C1/C2 compressors write FP4 codes
directly while retaining pre-RoPE latents for the indexer, avoiding the
fake-quantize/dequantize/repack route. C2 decode/verify share a writer with
draft-length-aware state handling. C4/C128 writers, decode views, prefill
dequantization, pool sizing and supported host-cache transfers carry the format
through the rest of the runtime. See the storage table above for current
defaults and restrictions.

**Reviewed files.** Under `python/sglang/`: JIT C++
`deepseek_v4/{c1,c2,fused_norm_rope_v2,main_norm_rope,store}.cuh` and
`include/sgl_kernel/deepseek_v4/kv_layout.cuh`; kernel wrappers
`ops/attention/dsv4/{attn,c1,c2,compress,dequant_k_cache,elementwise,kv_layout}.py`;
`srt/environ.py`, NPU `dsv4_memory_pool.py`, attention
`deepseek_v4_backend.py`, `dsv4/{compressor_v2,torch_quant}.py`, memory-cache
`{deepseek_v4_memory_pool,dsv41_request_window,kv_cache_configurator}.py` and
`hybrid_cache/hybrid_pool_assembler.py`. Tests under `test/registered/`:
`attention/unittests/dsv4/test_dsv41_fused_compress.py` and
`kernels/ops/attention/dsv4/test_{c2_verify,v41_kv_dequant,v41_kv_pool,v41_kv_store}.py`.

**Validation implications.** Check bytes against the format's reference
quantizer for store-only paths, and preserve norm/reduction tolerances for
norm-fused paths. Include padded two-token pages, untouched slots, skipped
graph rows, RoPE positions, compression state and dequant round trips. Changing
quantization semantics is distinct from removing redundant conversions.

### #39305: exact BF16 consumer top-k adapted from DeepSelect

[PR: Exact bf16 consumer top-k adapted from DeepSelect](https://github.com/sgl-project/sglang/pull/39305)
— merged. [Reviewed head](https://github.com/sgl-project/sglang/commit/97c2ed1e04a6e7dc0a98ff2197ad335168e1d58d).

**Motivation.** Candidate consumer rows fit in registers. The previous
FP16-derived 13-bit threshold histogram loses BF16 distinctions and uses a
larger histogram/block than needed for this shape.

**Implementation.** The SIMT adaptation uses one 512-thread CTA per row, up
to 32 BF16 scores per thread, and two 256-bin radix passes: high byte, then
low byte within the pivot bucket. It finds the threshold from the original
BF16 bits. Histograms use raw bytes; ordering is corrected once per lane during
pivot search rather than per score. A 16-word displacement of the negative
half reduces shared-memory bank conflicts. Per-thread greater/equal masks,
warp counts and tie quotas compact selected indices before a fused page-table
transform. The resource bounds in
[`TopKBF16Config`](https://github.com/sgl-project/sglang/blob/97c2ed1e04a6e7dc0a98ff2197ad335168e1d58d/python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh)
are:

```cpp
static constexpr uint32_t kVecSize = 8;
static constexpr uint32_t kMaxVecs = 4;
static constexpr uint32_t kElemsPerThread = kVecSize * kMaxVecs;
static constexpr uint32_t kMaxSeqLen = kBlockSize * kElemsPerThread;
static constexpr uint32_t kMaxTopK = 2048;
static constexpr uint32_t kNumBins = 256;
```

This implementation was tuned for 16384 scores and k=512; it is not a general
replacement for every top-k or the import of the whole DeepSelect package.
It returns unsorted indices and permits arbitrary choices among tied scores.
Signed-zero quota handling reconciles bit ordering with numerical equality.
NaN rows are not ordinary exact-top-k inputs: positive NaNs can consume quota
without yielding selected indices, leaving `-1` slots.

**Reviewed files.** `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh`,
`python/sglang/kernels/ops/attention/dsv4/topk.py`, and
`test/registered/kernels/ops/attention/test_topk_bf16.py`.

**Validation implications.** Compare selected values/sets rather than sorted
slot order. Cover odd row widths, ties, signed zeros, subnormals, infinities,
BF16 bit-pattern boundaries, page mapping and initialized padding. Do not
interpret NaN padding safety as a promise to rank arbitrary NaN inputs.
