# TokenSpeed selected-slot attention on tensor cores

Inspected on 2026-11-08. The complete PR diff, body, all comments and reviews
were read, then checked against current callers and provider registration at
TokenSpeed `7b21034e72bb45d1edef3d3b5483b50b58c4c492`. A fresh API check
matched the head/base below. Historical GPU results were not rerun here.

## PR #1780 — Compute selected-slot attention in absorbed-MLA tensor-core tiles

- Link: https://github.com/lightseekorg/tokenspeed/pull/1780
- State: merged 2026-10-23; head `7544be75dad9ee13a7a52b8c35ade4051ef5e924`,
  base `faa148c3ea9a38c8aad89e7a25041545836b93ce`,
  merge `85325280e03131ec83b174cb295c12c05967a164`.
- Coverage: all 967 diff lines, three files, +544/-294; three issue comments,
  two dismissed earlier-head reviews and no inline comments.
- Reviewed files: implementation and contract in
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/{triton.py,README.md}`;
  regression tests in `tokenspeed-kernel/test/ops/test_attention_dsa.py`.
- Motivation: the old portable kernel launches one program per token, head
  and value block. It computes scores with FP32 elementwise reductions,
  recomputes scores in a second pass, and repeats KV gathers across heads and
  value blocks. The author's GLM-5.3-Flash trace attributes 73.8% of summed
  kernel time across eight ranks to it; that sum is not a wall-time fraction.

The replacement processes 16 heads and 32 selected rows per tile. Absorbed
MLA uses the same latent vector for the non-RoPE key and value, so a gathered
latent tile feeds both QK and PV tensor-core operations. Online softmax
eliminates the separate maximum/accumulation passes. Probabilities and values
are BF16; accumulation and softmax state are FP32.

Packed FP8 rows contain latent bytes, an FP32 scale per 128 latent elements
and BF16 RoPE keys. After dequantization, scoring retains a second BF16 tile
for the rounding residual; PV uses only the high tile. Dense FP16/FP32 keys
likewise preserve latent and RoPE residuals for scoring. This improves LSE
accuracy without making the output bitwise equal to the old FP32 arithmetic.

```diff
+        latent_hi = latent.to(tl.bfloat16)
+        latent_lo = (latent - latent_hi.to(tl.float32)).to(tl.bfloat16)
+        scores = tl.dot(q_nope, tl.trans(latent_lo))
+        scores = tl.dot(q_nope, tl.trans(latent_hi), scores)
+        latent = latent_hi
```

Small grids split selected keys and allocate FP32 partial output/LSE buffers.
The power-of-two split count depends on `tokens * ceil(heads / 16)`, SM count
and selected-slot capacity; it is not a fixed decode/prefill switch. Large
grids use one split and omit the merge. Internal LSE is base 2; both paths
convert the public FP32 LSE to natural-log units. Empty splits have zero
weight, and empty or all-invalid selections produce zero output and `-inf`
LSE. Device lengths are clamped to `[0, slot_width]`; negative slots within
the live prefix are masked. Dense cache reads preserve page/token/channel
strides, and copying to a caller-supplied strided output updates that buffer.

### Current dispatch and subsequent changes

The portable decode/prefill registrations remain in `ops/attention/dsa/triton.py`
for NVIDIA and AMD. They admit BF16/FP8 E4M3 queries, latent ranks 128/512,
RoPE dimensions 0/64, page size 64, and selected widths 512/1024/2048–2051;
decode admits 1–8 query rows per request. The original non-RoPE dimensions
are 128/192/256. Other shapes or providers need their own evidence. Neither
registration advertises batch-invariant arithmetic or sorted slot order:
the current facade accepts this provider for `slot_order="selection"` and
rejects its use for `"sorted"`. Batch-invariant runtime plans select a provider
with the required feature rather than inheriting this split policy.

The paged DSA runtime passes the dense KV buffer with `sparse_kv_cache=None`,
casts queries to FP8 when the cache dtype is FP8, and passes the configured
solution/plan/order to the facade. KPool can extend a 2048-slot selection to
2051 slots. Under KVP it compacts owned slots, gathers query heads when
necessary and combines local output with natural-log LSE. The kernel ignores
`kv_seq_lens`; selected IDs and their live prefix must already encode valid,
causally visible slots. Supplying physical lengths alone does not add a
causal mask to this provider.

Main now factors dense scoring into `_dsa_dense_kv_attend`, shared with a
separate `dsa_prefill_dcp_records` core. That later core reads rank-major query
heads and writes BF16 output pairs plus FP32 LSE directly into exchange
records. Runtime admission requires the ordinary unplanned prefill selection
to resolve to this Triton provider, an FP8 cache, head-sharded KVP and an
unsplit launch. Its removal of relayout/repacking is not part of #1780, and
cannot be inferred for every prefill or batch-invariant plan. Use the logged
leaf and facade's shape capture to identify the actual provider.

### Validation and performance limits

The added test forces one- and four-split launches independently of GPU SM
count. Its two cases cover FP16 dense KV with BF16 queries and packed FP8 KV
with FP8 queries, 17 heads, width 2051, zero lengths, negative slots and an
all-invalid positive-length row. It compares output at `atol=rtol=5e-3` and
natural-log LSE at `1e-5`, checks strided supplied-output identity, exercises
decode and prefill, then changes lengths to negative/over-capacity values
before graph replay. A second test warms power-of-two batch sizes and checks
that irregular batches reuse compiled main/merge variants. These are defined
regression cases, not GPU tests executed by this refresh.

The body reports cold-L2 graph measurements on one H20, 2048 selected rows,
20 samples per case. FP8 KV B1/MTP4/8K falls from 529.89 to 25.82 microseconds;
BF16 KV falls from 582.30 to 21.97. It also reports eight-H20 TP8/EP8/MTP3
GLM-5.3-Flash AIPerf measurements with warmups excluded: 1K/1K/C1 TPOT
4.09→2.41 ms and C8 12.12→3.54 ms. Two paired passes show similar E2E changes
for C1, C8 and 32K prefill. These results compare the historical portable
baseline and candidate, not a current fastest-provider comparison.

Both variants reportedly score 15/16 on a small GSM8K subset with the same
miss and no truncation; this is not full-model accuracy certification. The
later conflict-resolution comment says main's paged in-place stride handling
and stride tests were retained, but no GPU tests were run on that resolved
head outside CI. The final-head AMD benchmark comment reports 579 cases
within budget. Its workflow logs were not inspected here, and the earlier
H20 performance/accuracy body does not pin the final rebased SHA. Preserve
that distinction when citing results.

Current source:
[kernel, split policy and registration](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/triton.py),
[facade and slot-order contract](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/__init__.py),
[paged runtime and KVP admission](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/attention/backends/paged/dsa.py).
