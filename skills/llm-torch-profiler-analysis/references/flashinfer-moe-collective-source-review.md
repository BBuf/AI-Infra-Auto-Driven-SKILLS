# FlashInfer MoE reduction, all-reduce and RMSNorm

Reviewed on 2026-11-09 (Asia/Shanghai). Historical code is pinned to the
final PR head below; current dispatch comparisons use FlashInfer
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU experiment was run for
this review. The public reduction API consumes dense expert-major rows;
the separate MoE-finalize API consumes a different producer contract.

## PR #4730 — Source-built Blackwell MoE all-reduce

- Link: https://github.com/flashinfer-ai/flashinfer/pull/4730
- State: merged on 2026-09-16; head
  `7fecac3e27446e8c6996c5c2c3b9f0794400a309`, base/merge-base
  `ed5f7b60eaa5b771f1cb986290d1515964694fc9`, merge
  `46afca6a173c239aa33a3a087ec181efcab237c7`.
- Coverage: all ten files, 16,024 diff lines, +15,863/-10; body, all
  20 issue comments, five reviews, seven inline comments and six commits.
  The 14,142-line generated CUDA file was split into all 28 complete
  entrypoints. Review covered the canonical FP16 body, every dtype/world-size
  and schedule difference, and both distinct optional-output store bodies.
  Exact comparisons verified the otherwise duplicated pairs; expanded polling
  predicates were checked against every low/high halfword in their arrays.
- Files: generated CUDA and `manifest.json` under
  `csrc/cake_trtllm_moe_allreduce_fusion/`; loader
  `flashinfer/jit/cake_trtllm_moe_allreduce.py`; API
  `flashinfer/comm/trtllm_ar.py`; the three
  `tests/comm/test_cake_*moe_allreduce*` files; incidental changes to
  `.pre-commit-config.yaml`, `csrc/trtllm_alltoall_prepare.cu` and
  `tests/comm/test_pcie_ipc_workspace_layout.py`.
- Motivation: combine expert weighting/reduction, shared-token addition,
  cross-rank summation, residual addition and RMSNorm in one selected kernel.
  The opt-in `backend="cake"` preserves the public default `"trtllm"`.

### Admission and numerical contract

The Python validator requires SM100/SM103, FP16 or BF16, world size 2/4/8,
rank in range, positive tokens/experts and hidden width 7168. Tensors must be
contiguous and on the same CUDA device. Expert input has `E * T * 7168`
elements in `[E, T, H]` order, route scales are FP32 `[E, T]`, and token,
residual and output tensors have `T * H` elements. Gamma has `H` elements.
These checks validate element counts, not a particular tensor rank.

Residual and norm outputs are **required**, despite the body's broader
optional-output wording. Only `moe_allreduce_out` is optional. Quant output,
scale output and quant layout must be absent; the API's `scale_factor` has
no numerical effect on these non-quantizing kernels. `weight_bias` is added
to gamma before its multiplication, with `None` mapped to zero.

The contiguous Int64 workspace table must contain at least `3 * W + 1`
pointers. The common byte guard is `T * H * 2 * W <= 2,145,386,496`.
It is not an inspection of the actual allocations behind those pointers.
The caller must retain compatible backing storage, capacity and process-group
ownership, and serialize uses of the mutable workspace. The C++ shim checks
device, supported activation dtype, world size and hidden width, but does not
repeat all Python shape, contiguity and scalar checks. Calling its FFI export
directly is not equivalent to calling the validated public API.

The mathematical order is observable:

1. Multiply each expert value by its FP32 route scale, round the contribution
   to activation dtype, then round after each expert addition. Add the shared
   token input with another activation-dtype rounding.
2. Publish that local result and sum rank contributions with rounding after
   every addition. TP4/TP8 use rank order 0 through `W-1`; TP2 starts at the
   local rank. FP16 uses packed `add.f16x2`; BF16 converts to FP32, adds, then
   explicitly rounds back to BF16 at each boundary.
3. Add the residual, round to activation dtype and write `residual_out`.
   Compute the squared norm from this rounded residual in FP32, multiply by
   `rsqrt(mean(square) + epsilon)` and `(gamma + weight_bias)`, then round the
   norm output. Optional all-reduce output is written before residual addition.

A single FP32 sum with one final cast is a different reference. Bounded-input
tolerance tests do not prove bitwise determinism, NaN/Inf handling or equivalent
model tokens. The generated file is compiled with `--use_fast_math`.

### Kernel schedule and workspace lifetime

Each token occupies four clustered CTAs of 224 threads, with eight values per
thread: `4 * 224 * 8 = 7168`. The launch grid is bounded by the SM count and
`4 * T`, rounded down to a multiple of four. Persistent clusters stride over
tokens. The kernel publishes its assigned local expert reductions to every
peer, clears an older Lamport generation, polls its local rank contributions,
and performs the residual/norm epilogue. Warp and cluster reductions combine
the norm statistic; cluster barriers protect its shared-memory exchanges.

The table's third world-sized region contains Lamport pointers, followed by
the local five-Int32 control allocation at index `3 * W`. Control words hold
the entry counter, phase, per-generation byte stride and previous clear size.
The workspace creator initializes three Lamport generations to negative-zero
sentinels. The kernel canonicalizes data zeros to positive zero before peer
publication, and polls every 16-bit lane against `0x8000`.

The phase cycles modulo three; a call uses `phase % 3` and clears
`(phase + 2) % 3`. At entry each CTA increments the counter after loading
control values. At the tail, CTA zero waits for that counter to reach the
grid size, advances the phase, records the next clear size and resets the
counter. This is an **entry/read counter**, not proof that all CTAs have
finished their output stores. Preserve the stream and workspace lifetime;
do not repurpose the counter as a generic completion event.

The manifest also describes TP4/TP8 acknowledgment tails. Those JSON fields
do not establish an active acknowledgment protocol: the actual allocator
reserves `W * 256 * 4` flag bytes, and these generated entrypoints address
the Lamport region and local control, without reading that flag region.
Inspect executed pointer accesses before attributing stalls or tuning a
purported acknowledgment tail.

There are 12 general dtype/world/output combinations, 12 SM103 single-token
variants and four SM100 TP8 mid-size variants. The latter apply exactly at
64 or 128 tokens, not an arbitrary interval. A representative exact diff:

```diff
+            float expert_scale_lane = 0.0f;
+            if (lane == 0) {
+                expert_scale_lane = expert_scales[expert * tokens + token];
+            }
+            float _shfl_0 = __shfl_sync(0xFFFFFFFF, expert_scale_lane, 0);
+            float expert_scale = _shfl_0;
```

This SM103 specialization broadcasts one lane's scale load and packs each
local result once before peer stores. SM100 TP8 at 64/128 tokens shares the
pack-once change, without the scale-load broadcast. Optional-output pairs
add only the corresponding eight-value all-reduce store. All other rounding,
Lamport and norm operations remain in the compared bodies.

The host launches on the current TVM stream, with 256 bytes of dynamic shared
memory and the optional PDL launch attribute. Device code waits for grid
dependencies before reading inputs/control and signals dependents at its
tail. Attribute presence alone does not establish useful producer/consumer
overlap. The source loader verifies manifest/hash/symbol inventory, caches by
source, host code, architecture and nvcc identity/version, locks compilation
and atomically publishes its cubin. It is not an added AOT registration.

### Tests, discussion and performance evidence

Six distributed pytest cases span two dtypes and TP2/4/8. Each tests 16
combinations: tokens 1/64/2048, eager/graph, PDL off/on, with both output modes
only at 64 tokens. That is 96 combinations in the committed harness, not
coverage of every physical specialization. In particular, token 128 is absent
and SM103 single-token kernels without the optional output are not exercised.
Inputs use eight experts, bounded random values and epsilon `1e-5`.

The reference retains contribution/addition/residual rounding. Distributed
checks aggregate failures and maximum absolute error across ranks, using
`atol=rtol=0.01` and rejecting NaNs. TP8 additionally reads the public control
phase on every rank and verifies `0,1,2,0`, poisoning outputs before those
four calls. The suite reuses one workspace across token sizes and modes.
Graph replays use unchanged inputs; the optional replay-count and rank-delay
arguments are not expanded by the test matrix. There is no deliberate
slow-rank campaign, concurrent-workspace test or model consumer in this diff.

The other twelve cases are CPU API/source contracts. Several inspect signatures
or generated strings; fake tensors and a mocked backend test dispatch without
executing the distributed math. The nvcc cache-version tests exercise actual
cache-key behavior. Inventory counts must not be presented as GPU validation.

Fresh final-head [CI run 35057314766](https://github.com/flashinfer-ai/flashinfer/actions/runs/35057314766)
is successful. Its [H100](https://github.com/flashinfer-ai/flashinfer/actions/runs/35057314766/job/104670165861),
[T4](https://github.com/flashinfer-ai/flashinfer/actions/runs/35057314766/job/104670165841)
and [A10G](https://github.com/flashinfer-ai/flashinfer/actions/runs/35057314766/job/104670165812)
logs each show **12 passed, six skipped**: all distributed cases are skipped
on these unsupported architectures. Four AOT import jobs also pass; this does
not compile/run the opt-in Blackwell kernel. The targeted run covers the three
Cake test files, not all communication regressions. Preserve the body's older
T4 repair and unresolved A10G all-to-all/cuDNN failure reports as historical
evidence; this narrower green run does not resolve those earlier failures.

The incidental T4 fix constructs shared tile scratch and the cooperative block
partition before inactive receive tiles return. The receiver runs in the
1024-thread count/indices kernel. The PCIe test change adds SM120 TP4/8 binary
flag bytes to exact workspace expectations. Neither change widens the Cake
backend to T4 or SM120. The author's paired T4/H100 eight-case reports are
separate from the final targeted run; the T4 synccheck timeout and racecheck
hazards are not a clean sanitizer pass.

Most review comments discuss superseded broader files, including a finalize
export and benchmark no longer in this ten-file diff. Final code keeps the
reduction export only, validates shim dtype/world/hidden width and includes
nvcc version in its cache key. Its build dependency requires TVM FFI >=0.1.11,
above the review's requested >=0.1.10. The two maintainer approvals belong to
`78b8683b6903` and `2e4f80320354`, before the final merge of main. The last bot
comment links a running GitLab pipeline; it is not itself a completion result.

The body reports candidate `cc8a5d2cc35c7ad3e078e34e4a494b9ffb09ac72`
against integration parent `f4aeb3706efda8ba064d5d64dfdc59fb4d9a8d48`:
20 B200 rows at geometric mean **1.122930x**, and 12 B300 rows at **1.121959x**.
B200 geometric-mean latency is 21.321571 to 18.987444 microseconds.
These are native operator reports with cold-L2 CUPTI timing and correctness
checks before/after. Accepted receipts were retained across attempts; final
attempts reran only selected interrupted/invalid rows. This is not a fresh
32-row campaign on every publication or a serving throughput measurement.
The final diff contains no benchmark harness reproducing the external receipts.

Local byte comparison confirms that candidate and final head have identical
kernel, manifest, loader, API/validator, API tests and distributed tests. The
source-inventory test differs: its stale 12-symbol/13-launch expectations were
corrected to 28 symbols/29 macro occurrences. Runtime byte identity connects
the reported candidate to this publication, but does not independently reproduce
timing, compiler/environment or numerical results. The merged-library report
also does not turn closed [SGLang #39214](sglang-moe-collective-source-review.md)
into a mainline consumer or prove its serving speedup.

### Current dispatch and profiling guidance

Frozen main has replaced this monolithic loader with
`cake_trtllm_moe_allreduce_union.py` and individual kernel translation units.
The public Python validator is unchanged, but current specialization keys
include architecture, world size, dtype, PDL, expert count and inclusive token
ranges. For BF16, eight experts and PDL off, both SM100/103 now choose
`lat_t1f` at one token and `lat_t1` at 2–32 tokens. This is not the historical
SM103-only single-token rule. Generic/wide-MLP fallbacks and two-shot variants
have separate residency parameters and, where recorded, cluster caps.

Current kernels always write the all-reduce output. Passing `None` selects a
cached scratch sink, rather than the historical no-store specialization.
The cache grows geometrically and retains replaced allocations so existing
CUDA graphs keep valid addresses; allocations made during capture are returned
without entering that process cache. Account for these live/retired buffers
when comparing memory and optional-output timing. The current launcher checks
tensor dtype/device/contiguity and Int32 scalar ranges, verifies the current
device, and uses the runtime PDL/cluster launch attributes. This review reads
the complete current selector and launcher; it does not count every successor
PR or union device program as a completed historical review.

For a trace, record the exact library revision, selected symbol, expert count,
tokens, PDL and workspace owner. Include producer layout/gather/scale-conversion
costs when comparing a serving integration. The historical 1.12x cannot be
assigned to the current union or to SGLang's different CuTe-DSL boundary path.

Immutable source anchors:

- Historical [API and workspace](https://github.com/flashinfer-ai/flashinfer/blob/7fecac3e27446e8c6996c5c2c3b9f0794400a309/flashinfer/comm/trtllm_ar.py),
  [loader](https://github.com/flashinfer-ai/flashinfer/blob/7fecac3e27446e8c6996c5c2c3b9f0794400a309/flashinfer/jit/cake_trtllm_moe_allreduce.py),
  [all generated programs](https://github.com/flashinfer-ai/flashinfer/blob/7fecac3e27446e8c6996c5c2c3b9f0794400a309/csrc/cake_trtllm_moe_allreduce_fusion/cake_trtllm_moe_allreduce_fusion_kernels.cu)
  and [distributed tests](https://github.com/flashinfer-ai/flashinfer/blob/7fecac3e27446e8c6996c5c2c3b9f0794400a309/tests/comm/test_cake_moe_allreduce_distributed.py).
- Current [public API](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/comm/trtllm_ar.py),
  [selector and scratch lifetime](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/jit/cake_trtllm_moe_allreduce_union.py)
  and [launcher](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/csrc/cake_trtllm_moe_allreduce_union/cake_trtllm_moe_allreduce_union_launcher.cu).

## PR #5856 — Consolidate calls without an all-reduce output

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5856
- State: merged on 2026-10-01; head
  `8e42ce03a009d29a7b6d00a54b7ed6873c7bf8e5`, base/merge-base
  `122037aaa23f068578e8a741586de22f42abf27b`, merge
  `08776e742cf9b07154b36695947a7b6e11c3c29a`.
- Coverage: all nine changed files, 6,875 diff lines, +158/-6,458; body,
  nine issue comments, four reviews, three inline comments and the sole commit.
  The deleted 5,978-line CUDA file contains twelve complete kernel bodies,
  each byte-identical to a body already reviewed in #4730. All remaining
  preprocessor/header differences, the deleted 226-line launcher and 79-line
  loader, and every Python/test/config change were read. The distributed
  worker and current union selector/caller were also read.
- Files: `.pre-commit-config.yaml`; the two deleted `.cu` files under
  `csrc/cake_trtllm_moe_allreduce_fusion/`; deleted
  `flashinfer/jit/cake_trtllm_moe_allreduce.py`; public API
  `flashinfer/comm/trtllm_ar.py`; union loader; API, distributed and union
  tests under `tests/comm/`. The final diff does not delete a test file,
  despite the body's statement about deleting the structure test.

Previously, an omitted `moe_allreduce_out` selected the isolated no-store
bundle. Its six generic programs cover FP16/BF16 and TP2/4/8; six single-token
programs are compiled only for SM103. Its launcher uses four 224-thread CTAs
per cluster, an SM-bounded persistent grid, 256 bytes of shared memory and
the current CUDA stream. The refactor removes this separate loader and dispatch
branch. After the existing public validation, every Cake call reaches the
union; `route_applies` no longer depends on output presence. SM100/103, TP2/4/8,
hidden 7168, dtype/layout and quantization restrictions remain.

The union itself already always stores its all-reduce result. Its entire
`csrc/cake_trtllm_moe_allreduce_union/` tree is identical before and after
this PR; the new Python caller supplies the missing sink:

```diff
+    if moe_allreduce_out is None:
+        moe_allreduce_out = scratch_allreduce_output(
+            moe_reduction_active_experts_token_input.device,
+            moe_reduction_active_experts_token_input.dtype,
+            int(token_num),
+        )
```

That sink adds one `[T, 7168]` 16-bit output per rank: 14 KiB per token,
or 28 MiB at T=2048. The initial helper caches one allocation per device/dtype,
returns a smaller view when possible, and replaces it on growth. A fresh
allocation made during capture is returned without entering that cache.
This distinction left the warmed-cache graph-address bug fixed in #5894 below;
the original commit's general claim of capture safety was too broad. Current
main retains the unified route and uses the repaired lifetime rule.

The body reports 30 rows on each of eight-GPU B200 and B300 nodes with CUDA
13.1, comparing the old no-store bundle against union plus scratch. Reported
geometric means of **union/legacy latency** are 0.638 and 0.643 respectively;
the B300 TP8/BF16/T1 row regresses from 11.01 to 11.25 microseconds. These
compare different kernel schedules, so they do not isolate the added store's
cost. They also do not measure current union rules or model serving.

The reported protocol uses three blocks with alternating arm order, ten
warm-ups and thirty timed iterations per block, separate workspaces, and the
median of per-iteration maximum rank times from CUPTI. It disables the
per-rank L2 flush because its memset/host synchronization can skew arrivals
at Lamport polling; the body records an outlier under the flush-enabled
protocol. Preserve that cache/arrival regime when comparing the numbers.
For a new measurement, inspect rank arrival skew as well as kernel duration
and report the cache regime explicitly. No paired benchmark harness or raw
samples are added in the nine-file diff. The final rebase's entire non-config
diff is byte-identical to `09c6797f11493cf846ad0737b1067c426d31da21`, supporting
the author's code-identity statement, not independently reproducing its timing.

CPU tests check forwarding to the union, rejection before dispatch and scratch
reuse/growth/dtype separation. The six distributed test parameters are
TP2/4/8 times FP16/BF16; each runs many internal cases. Missing-output cases
now include T=1, 64 and 2048 with eight experts, PDL on/off and eager/graph
execution. The graph helper warms the same shape and destroys its local graph
before the next shape grows the cache, so this matrix does not test a surviving
old graph after growth. Residual/norm checks use the existing tolerance;
there is no exposed all-reduce tensor to compare in missing-output cases.

The author's B200/B300 report claims 69 CPU and six distributed passes per
architecture with `TORCH_SYMMMEM=NVSHMEM`. Inspected final-head
[CI run 36915429078](https://github.com/flashinfer-ai/flashinfer/actions/runs/36915429078)
is narrower: the [H100 log](https://github.com/flashinfer-ai/flashinfer/actions/runs/36915429078/job/110548297908)
checks out the final-head merge and reports **69 passed, six skipped**, with
all distributed cases skipped under CUDA 12.9. Six AOT jobs also pass. An
earlier GitLab bot report has 19/29 executed jobs passing and unresolved
GB300 jobs without reports; the later bot announcement only says running.
Do not turn either the H100 green result or that announcement into verified
Blackwell distributed coverage. The review's scratch-lifetime finding was
addressed in the separate #5894 fix, not in this refactor's final head.

Immutable [dispatch](https://github.com/flashinfer-ai/flashinfer/blob/8e42ce03a009d29a7b6d00a54b7ed6873c7bf8e5/flashinfer/comm/trtllm_ar.py),
[scratch and selection](https://github.com/flashinfer-ai/flashinfer/blob/8e42ce03a009d29a7b6d00a54b7ed6873c7bf8e5/flashinfer/jit/cake_trtllm_moe_allreduce_union.py),
[distributed tests](https://github.com/flashinfer-ai/flashinfer/blob/8e42ce03a009d29a7b6d00a54b7ed6873c7bf8e5/tests/comm/test_cake_moe_allreduce_distributed.py)
and [deleted kernel source at the base](https://github.com/flashinfer-ai/flashinfer/blob/122037aaa23f068578e8a741586de22f42abf27b/csrc/cake_trtllm_moe_allreduce_fusion/cake_trtllm_moe_allreduce_fusion_kernels.cu).

## PR #5894 — Retain scratch addresses used by captured graphs

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5894
- State: merged on 2026-10-01; head
  `2afcd7d7157f6c45586ffafb673a38386f433f3a`, base/merge-base
  `08776e742cf9b07154b36695947a7b6e11c3c29a`, merge
  `0c101ac509fa4a8b2ab8acdf1528c22432deb4fa`.
- Coverage: the full 137-line, two-file diff, +78/-8; body, all four issue
  comments, one approving review, no inline comments and the sole commit.
  Files are `flashinfer/jit/cake_trtllm_moe_allreduce_union.py` and
  `tests/comm/test_cake_moe_allreduce_union.py`. The current helper is
  source-identical to this PR head; the current caller/launcher were also read.
- Motivation: after a graph records an existing scratch allocation's address,
  an eager call needing more rows used to replace the cache's only ownership
  reference. Later replay could write into storage reused by an unrelated tensor.
  This follows the scratch-output path introduced earlier; it changes no
  kernel, launcher, output numerics or route-selection rule.

The trigger is `backend="cake"` with `moe_allreduce_out=None`, an already
large-enough cache at capture, then a larger eager request. If capture itself
needs a fresh allocation, that allocation belongs to its graph pool and is
not installed in the process cache. These two paths have different owners.

The patch keeps replaced tensors in a process-lifetime list and grows capacity
geometrically. Exact changed lines:

```diff
+        capacity = token_num if cached is None else max(token_num, 2 * cached.shape[0])
+        fresh = torch.empty((capacity, HIDDEN_DIM), dtype=dtype, device=device)
         if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
-            return fresh
+            return fresh[:token_num]
+        if cached is not None:
+            _retired_scratch_allreduce_outputs.append(cached)
         _scratch_allreduce_outputs[key] = cached = fresh
```

Views still expose only the requested rows. For each `(device, dtype)` cache,
every replacement at least doubles capacity, so the sum of retired capacities
is less than the live capacity. Live plus retired payload is consequently less
than twice the live tensor's bytes. For example, a live 2048-by-7168 BF16 sink
holds 28 MiB; that cache's live and retired payload together stays below
56 MiB. This bound excludes graph-private allocations, allocator overhead and
other device/dtype caches. It is not a bound on total communication workspace
or GPU memory. A smaller subsequent request does not shrink the cache, and
destroying a graph does not reclaim this process-lifetime retirement list.

The CPU test checks a four-row allocation, a two-row view, growth to capacity
eight for a five-row request, reuse at eight rows, retirement and dtype
separation. The GPU test captures a fill of a cached two-row view, exercises
a larger graph-private allocation without changing the cache, then eagerly
grows the cache and replays while checking a separate canary stays zero.
It also explicitly asserts that the recorded pointer is in the retired list.

That GPU test has an important limit: its `warm` and `captured` variables still
hold the old storage through growth and replay. The canary check therefore
does not independently recreate allocator reuse after dropping the last owner.
The retirement-list assertion checks the new ownership rule, while the GPU
operations check capture/replay compatibility. Neither is a negative-control
reproduction of the original corruption. A stronger reproduction must release
incidental tensor/view references before growth and establish actual address
reuse on the broken path; the existing test does not do that. It also fills
the scratch directly rather than invoking a multi-GPU all-reduce kernel.

The author's B200 report says 70 cases passed on the stated base plus this
change. Independently inspected final-head
[CI run 36930534914](https://github.com/flashinfer-ai/flashinfer/actions/runs/36930534914)
also passes: the [H100 log](https://github.com/flashinfer-ai/flashinfer/actions/runs/36930534914/job/110598554104)
checks out the merge of `2afcd7d7157f` into the recorded base and reports
**70 passed, zero skipped**, including the new capture test, under CUDA 12.9.
This test can run on H100 because it exercises tensor/graph lifetime without
launching the SM100/103 collective. The 70 cases are not 70 distributed
numerical validations. Six AOT import jobs pass separately. The discussion's
GitLab bot message only announces a running pipeline; the inspected GitHub
logs provide the completion evidence.

For profiling and capacity analysis, attribute these allocations to the
optional-output sink, separate from KV pools. Verify the deployed route and
cache growth history before diagnosing a leak or estimating reclaimable bytes.
Do not clear the cache/retired list as an OOM workaround while captured graphs
can still reference it. The PR contains no new latency/throughput benchmark;
its shared performance banner is not evidence of a speedup from this fix.

Immutable [helper and caller](https://github.com/flashinfer-ai/flashinfer/blob/2afcd7d7157f6c45586ffafb673a38386f433f3a/flashinfer/jit/cake_trtllm_moe_allreduce_union.py)
and [tests](https://github.com/flashinfer-ai/flashinfer/blob/2afcd7d7157f6c45586ffafb673a38386f433f3a/tests/comm/test_cake_moe_allreduce_union.py).
