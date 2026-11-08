# SGLang DeepGEMM MQA migration: precision, schedules and cache ownership

Reviewed on 2026-11-09 (Asia/Shanghai). No GPU capture, numerical test,
model evaluation or performance rerun was performed for this review.

## PR #44350 — migrate DSA/DSV4 indexers to DeepGEMM 0.2.5.post2

- Link/title: [[Refactor] [NVIDIA] attention: Migrate DSA/DSV4 indexers to DeepGEMM 0.2.5.post2](https://github.com/sgl-project/sglang/pull/44350).
- State: open, not merged, with a merge conflict at this check. Head
  `3b172a83a6361b45e478d2c8e32dde9ee1883079`; API base and merge base
  `f1a36c4789e6a4e766e00a902728ebd37a9df604`.
- Coverage: all 3,985 lines of the complete 49-file diff read, +1,801/-335;
  full body, all 23 commit messages and seven issue comments read. No formal
  reviews or inline comments were returned. Complete changed files were
  saved, and surrounding cache-key and compatibility paths inspected.
- Motivation: the replacement SM100 MQA interface requires MX scales,
  BF16 head weights/output and request-grouped varlen metadata. Merely
  renaming the calls would misinterpret cache bytes, schedule the wrong
  work, and give existing FP32 top-k consumers incompatible outputs.

This is a historical open migration, **not the current mainline upgrade
recipe**. The frozen SGLang main pin `c24285bde97c3e6424cd5d8cb35788889034d8ea`
already pins [sgl-deep-gemm 0.2.6](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/pyproject.toml)
and does not contain this proposed adapter file. Current DSA still resolves
the legacy `fp8_mqa_logits` / `fp8_paged_mqa_logits` functions. Current V4.1
[index-logits dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/index_logits.py)
admits the modern interface for BF16 SM100 weights/logits, but paged calls
also require varlen metadata with `indices`. Current
[metadata admission](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/metadata.py)
requires the varlen flag, a non-JIT schedule, at least 96 total query rows,
and at most 29,039 rows in a chunk. Below that minimum, or without those
indices, even BF16 paged output uses the legacy call shape. FP32 overrides
and other architectures also retain legacy calls. Output dtype alone does
not identify which scheduler or kernel ran.

The independently inspected DeepGEMM `dev`
[`909f68c19cc23c6afd9cc039c0493b00586b66b2`](https://github.com/sgl-project/DeepGEMM/blob/909f68c19cc23c6afd9cc039c0493b00586b66b2/deep_gemm/__init__.py)
has both interfaces: a boolean trailing argument or `clean_logits` /
`logits_dtype` keyword selects legacy code; SM100 metadata without `indices`
also selects the legacy builder. This source inspection does not establish
which binary is installed on a serving machine. Do not downgrade its wheel
to this PR's pin or impose the proposal's cache format based on the PR title.

## Adapter, numerical and memory contracts

The proposed [adapter](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/srt/layers/attention/deep_gemm_mqa_logits.py)
leaves SM90-style Q/scales alone. On CUDA major 10 it gives plain E4M3 Q
unit packed UE8M0 scales, passes already quantized Q tuples through, and
casts head weights to BF16. The original Q scale is already folded into
those weights. BF16 output widened to FP32 does not recover lost precision;
near-equal scores can change selected token membership.

Ragged output is compressed: column `j` of row `i` denotes key `ks[i]+j`.
Callers requesting the old full-width form receive a separate FP32 matrix
with `-inf` outside `[ks, ke)`. KPool instead consumes compressed rows and
removes the old top-k row-start offset on CUDA; its AITER path retains
full-width rows and offsets. The AITER 2 GiB chunk helper is relocated,
not evidence of a new AMD kernel speedup.

Paged FP32 consumers get a **prefix-only conversion**:

```diff
+    if logits_dtype == torch.float32:
+        return widen_logits_prefix(logits, context_lens.reshape(-1))
+    return logits.to(logits_dtype)
```

Only each row's live context is copied. The FP32 allocation still covers
the full padded width, and the source BF16 matrix remains live during the
copy; unwritten tails are not clean logits. Consumers must bound reads by
length. The adapter rejects paged `clean_logits=True`. V4.1 low-ratio FP4
paths honor `index_logits_dtype()` and already default to BF16 on SM100;
they do not all incur this conversion.

The [memory helpers](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/srt/layers/attention/mqa_logits_utils.py)
account for simultaneously live compressed and expanded buffers: roughly
six score bytes per element on SM100, versus eight for FP32 DeepGEMM
compressed-plus-expanded rows elsewhere, before row alignment and other
scratch. KPool and V4 C4 budgets similarly include BF16-to-FP32 overlap.
These are allocation budgets, not measured bandwidth. Paged decode paths
that widen are not covered by the prefill budget; graph capture can also
bypass eager budget checks. A shorter conversion kernel does not prove
lower peak allocation.

## Scale bytes, PD and external storage

Three explicit writer modes replace the ambiguous rounding boolean:
FP32, power-of-two FP32, and packed UE8M0. The shared
[C++ helper](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh)
rounds a positive non-power-of-two scale upward using its exponent and
nonzero mantissa, then uses the corresponding inverse for quantization.
Its older comment says “nearest”; the implementation is upward rounding.
POW2 stores the exponent as an FP32 power of two; UE8M0 replicates its
exponent byte four times for the four 32-channel groups. Both occupy four
bytes per 128-channel token, so shape/byte-size checks cannot distinguish them.

DSA chooses UE8M0 when its SM100 paged reader is DeepGEMM. KPool always uses
that reader on SM100, regardless of the ordinary DSA backend setting.
V4 C4 excludes TileLang, AITER and torch-fallback configurations from its
UE8M0 choice. These decisions are derived from runtime settings in several
places; this patch does not make the pool own a single immutable format field.

The adapter's `pack_ue8m0_x4` only extracts exponent bits. It is lossless for
the documented power-of-two inputs, not a general FP32-to-UE8M0 conversion:
for example, a positive scale of 1.5 becomes an exponent encoding of 1.
The PD tag `index_k:fp32` includes both ordinary FP32 and POW2, so a tag match
alone does not prove that remotely produced scales meet the adapter's
power-of-two precondition. The new writer tests compare POW2 and UE8M0;
they do not establish arbitrary-FP32 peer compatibility.

The [PD changes](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/srt/disaggregation/common/conn.py)
send `decode_kv_layout` even when empty. Python and Rust bootstrap handlers
reject a UE8M0 prefill when the query lacks that field. For new decoders,
`pd_kv_layouts_match` rejects unequal explicit layouts and allows a missing
layout only when neither side names UE8M0. These checks do not convert pages;
SM100 UE8M0 and SM90/SM120 FP32 deployments can therefore be incompatible.

The [October 31 review comment](https://github.com/sgl-project/sglang/pull/44350#issuecomment-6536347767)
raises a separate old-decoder problem. The inspected pre-layout
[`PrefillServerInfo`](https://github.com/sgl-project/sglang/blob/834bfe4ef0ec11ac5625fac3f94f60c041a36cb8/python/sglang/srt/disaggregation/common/conn.py)
has no `kv_layout` constructor field. Even an FP32 prefill's new explicit
field can fail that decoder's `PrefillServerInfo(**data)` construction.
The PR's mixed-version tests exercise current handlers with missing query
fields; they do not run that historical decoder.

HiCache's tagged `model_name`, Mooncake/MORI direct-linker names, and a
copied LMCache `model_config.model_path` separate UE8M0 namespaces where
those values participate in keys. The original model config is preserved.
This is not universal L3 protection. At the reviewed head, HF3FS's
[`_pool_component_key`](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py)
uses the incoming page key plus pool/rank fields, without this model tag;
FlexKV receives its original model configuration through a separate path.
The unified key scheme can incorporate model identity in its namespace
digest, so inspect the selected backend and key scheme together.
Do not turn the reviewer's entire backend list into a demonstrated
corruption claim: AIBrix rejects MLA, while the reviewed `shm` backend is
a local dummy store that returns misses and `mmap` is an allocator.

## Schedule grouping and graph addresses

On SM100, metadata and kernel calls must use the same request grouping.
The adapter flattens `[batch, next_n]` into rows, repeats block-table rows
and supplies `indices = arange(batch * next_n) // next_n`. Building grouped
metadata but launching independent request rows is not an interchangeable
schedule. CuTe DSL retains its own 256-token split metadata; modern SM100
DeepGEMM uses 384-token splits, and SM120 uses 128. Draft-extend CuTe DSL
uses one query-token row per request and rebuilds its matching schedule.

[DSV4 metadata](https://github.com/sgl-project/sglang/blob/3b172a83a6361b45e478d2c8e32dde9ee1883079/python/sglang/srt/layers/attention/dsv4/metadata.py)
caps SM100 DeepGEMM calls at 16,384 rows and SM120 at 4,096, further reducing
chunks to respect the memory budget and request-group multiple. Ordinary
DSA's SM100 schedule remains unchunked, retaining the reported upstream
shared-memory limit of roughly 29k rows. The generic JIT metadata cannot
silently replace an incompatible DeepGEMM schedule at large batch sizes.

For equally chunked graph metadata, schedules and top-k plans are copied
into existing tensors:

```diff
+        if isinstance(dst_val, list):
+            assert len(dst_val) == len(src_val), f"{field_name=} chunk count differs"
+            for dst_item, src_item in zip(dst_val, src_val):
+                dst_item.copy_(src_item)
```

Rebinding a Python list would leave captured kernels reading old addresses.
The different-chunk path still rebinds; this patch does not establish replay
compatibility across changed chunk topology. Dense/paged FP4 dispatch remains
available on SM120, while the sparse-table candidate path is SM100-only and
uses the dense candidate fallback elsewhere.

## Tests, CI and performance evidence

The diff adds real cache-byte, ragged-column, nonidentity-page, backend
schedule and namespace tests. CPU metadata cases check same-object copies
for changed schedule/top-k contents and mocked SM120 dispatch. These are
useful behavior checks, not SM120 GPU execution. The SM100 score test changes
from per-element tolerance to `calc_diff < 3e-5` with BF16-rounded weights;
the sparse-selection test compares sorted selected scores at ties. Neither
establishes exact FP32 score or token-index parity. The metadata-reference
test explicitly skips SM100's incompatible format.

Fresh paginated queries for the exact head returned 23 workflow runs,
139 check runs and no commit statuses. Base
[40985816538](https://github.com/sgl-project/sglang/actions/runs/40985816538),
extra [40985816290](https://github.com/sgl-project/sglang/actions/runs/40985816290)
and AMD [40985816504](https://github.com/sgl-project/sglang/actions/runs/40985816504)
all fail the `run-ci` label gate. Their logs confirm that reason; the
relevant CPU/JIT/GPU jobs are skipped. Successful title/lint or unrelated
workflow checks do not validate the numerical migration.

The following remain **author-reported historical runs**, not final-head CI:

- B300/SM103, torch 2.14.1+cu130, DeepGEMM 0.2.5.post1 at `56b3fabdee`:
  six suites, 632 passes, 18 skips and 40 subtests. Earlier `dabf3a9ce9`
  tables cover more cache/indexer suites. The 24,576-row, two-chunk graph
  address check and 72 grouped-versus-per-row bitwise cases are ad hoc
  reports. No SM120 hardware run is reported.
- H200 results use an earlier custom wheel; the paged-logit tolerance
  failure reportedly also occurs on its old-main control. Four HiCache
  lifecycle failures and a missing tensorcast dependency likewise remain
  recorded as environment/control failures, not passes.
- V4-Flash TP4 on four B300s used the low-latency recipe,
  `flashinfer_mxfp4`, EAGLE 3/1/4, GSM8K 1,319 questions and AIME25
  16 samples/problem with a 64k token cap. On base `e7c55f6959`, GSM8K is
  0.968 in both arms; AIME25 is 0.835 versus 0.848, both below the harness's
  0.93 threshold. The 0.935 result uses an older base. The author links the
  base-dependent difference to reasoning-effort/prompt changes; that
  separate prompt PR has not been fully reviewed here. Do not attribute
  the cross-base difference to this kernel migration.
- Serving random 128-in/128-out, concurrency 256, 2,048 requests, warmup
  and ABBA order: main 3,866/3,887 output tokens/s versus PR `5a4b7e7097`
  3,850/3,850, mean TPOT 60.9/60.6 versus 61.1/61.1 ms. Same-code boot
  variation was 0.6–1.8%; this supports no measured service gain. The
  earlier whole-width conversion's 799 us reportedly fell to 38 us with
  prefix conversion, removing an earlier approximately 7% regression.
- DSA verify at batch 1/context 32k/next-n 4: mismatched schedule/call
  143.9 us, grouped 25.8 us, correctly matched independent rows 24.4 us.
  The five reported shapes include grouped wins and losses against the
  properly matched control. A separate V4 C4 grouping experiment improved
  its kernel 22–56% but was flat in serving and is explicitly deferred.

No DSA-model, PD-pair or external-L3 end-to-end run, and no rerun on the
final 0.2.5.post2 pin, is reported. Those limits matter when recommending
the migration rather than only interpreting a trace.

## Reviewed files and implications for LiteTopK

All 49 files are covered: the two dependency pins; three C++ cache writers
and shared FP8 helpers; DSA/DSV4 Python store, compression and logits wrappers;
the adapter; DSA, KPool, C4 and V4.1 callers; schedule/metadata/budget code;
Python/Rust PD bootstrap; cache controller, memory pool and three direct
storage integrations; the paged-MQA test helper; two benchmark files; and
the changed kernel/unit tests for logits, cache writes, chunking, graph
copies, PD layouts and LMCache configuration.

The [LiteTopK histogram producer](deepgemm-litetopk-producer-review.md)
still materializes BF16 scores. Its SGLang consumer was stacked on an older
version of this migration, whose ordinary path widened the entire score
width. Savings against that old baseline include removed conversion work;
they must be remeasured against prefix-only conversion and current BF16
V4.1 dispatch. This review completes the migration dependency; the full
SGLang LiteTopK consumer reconciliation remains in progress.
