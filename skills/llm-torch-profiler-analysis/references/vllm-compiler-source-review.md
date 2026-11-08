# vLLM RoPE/cache fusion reachability and retained operations

Inspected on 2026-11-09 (Asia/Shanghai), against the November 8 frozen vLLM
main `b022bb7a4873a47cdcdf474edf3bd499c221f40d`. All four complete final PR diffs,
bodies and all discussion were read; fresh API checks still match the heads
below. Current configuration ordering, pass registration, all three cache
fusion files, backend methods and relevant configuration tests were inspected
separately. For the capture-binding proposal, the complete current DSV3.2
attention class, fused norm/RoPE kernel and wrapper, and relevant capture/slot
buffer paths were also read. This review does not reproduce the reported GPU
experiments.

## Current main: registration is not proof of a fused kernel

The ordinary RoPE/cache and combined QK-norm/cache flags are ROCm-gated in
`PassConfig`; the MLA concat/cache flag has a CUDA-alike gate. In the normal
Dynamo splitting path, `splitting_ops=None` adds attention and both cache
update ops to the split list. It disables explicitly requested ordinary and
QK-norm/cache fusion. An explicit MLA flag is not cleared in that branch,
but its cache-update op is still split out, so the flag alone does not prove
the pattern can match. Attention-output-quant and Inductor partition paths
have different splitting rules and must be checked separately.

There is also an ordering dependency: `CompilationConfig.__post_init__`
adds `+rotary_embedding` for already-enabled ordinary/QK fusion flags before
`VllmConfig` resolves optimization-level defaults. The current ordinary
resolver requires AITER, that custom op already enabled, and cache-update
visibility. The QK resolver requires AITER and the custom op but lacks the
visibility check. Thus O2/O3 entries in the defaults table are conditional
resolvers, not a promise that these fusions turn on. Main computes compile
ranges before splitting can turn a flag off; stale endpoints can remain.

`_USE_LAYERNAME` requires the PyTorch opaque-type support check
(`>=2.11.0.dev`) and `VLLM_USE_LAYERNAME`. It wraps names as hoisted graph
inputs, avoiding a distinct compiled graph for each constant layer name.
This also lets one registered pattern serve homogeneous attention layers.
Removing all splitting ops is a separate capture-policy change: the current
code turns PIECEWISE into NONE and FULL_AND_PIECEWISE into FULL. Do not offer
`splitting_ops: []` as an isolated kernel toggle.

The matched graph and actual GPU replacement have different boundaries:

- `RopeStaticQQuantKVCachePattern` fuses RoPE/cache but still calls
  `_C.static_scaled_fp8_quant` on Q afterward. A remaining Q-quant kernel is
  expected, even when the pattern matches successfully.
- `QkNormRopeKvCachePattern` and its MRoPE variant also reapply Q quantization
  after their fused norm/RoPE/cache op. Per-tensor AITER quant preserves the
  updated scale output; the generic static form preserves its group shape.
  The replacement does not fuse Q quantization into the cache writer.
- The combined pass accepts BF16/FP16, supported backends, head dimensions
  64/128/256 and equal Q/K/V head dimensions. It sets interleaved V-cache
  state on every eligible layer, even when a hoisted pattern is registered
  only once. The ordinary `ROCM_ATTN` RoPE/cache method falls back to separate
  rotary and cache updates when it encounters that interleaved layout.
- MRoPE additionally validates three positive sections, rotary extent and
  style-dependent alignment. Its fused writer does not return rotated BF16 K;
  the matcher rejects other observable users of the returned raw K view,
  allowing the decoder attention consumer that reads the updated cache.
- Ordinary and combined QK fusion apply only when the compile-range end is
  at most `rope_kvcache_fusion_max_token_num` (default 256). MLA is a separate
  pass without this range override: its op calls
  `prepare_kv_cache_update(attn_metadata)` before `concat_and_cache_mla_rope_fused`.

The pass manager runs split/scatter cleanup before the combined QK pass and
ordinary RoPE/cache pass. Muse-specific registration has its own gates; these
generic flags do not describe every model-specific prologue fusion.

The separate [CUDA combined-kernel review](vllm-qknorm-rope-cache-source-review.md)
covers open #57682's proposed FlashAttention hook, homogeneous-layer gate,
stacked partial-RoPE fix and old-wheel validation limits. It remains absent
from this pinned main; its direct 2,048-token compile test does not exercise
the normal pass manager's 256-token range gate.

## PR #57349 — Honor explicit fusion requests with hoisted layer names

- Link: https://github.com/vllm-project/vllm/pull/57349
- State: closed without merge on 2026-11-03; head
  `b3d822b49f43fdc7f5292cf0002f30bcd223ead6`, base
  `cd9aa5b5ee4b8b2e035517971e0d5e184baa787f`. The author's closing comment
  says it was folded into #57900; it is not a second active implementation.
- Coverage: all 216 diff lines, four files, +101/-29; body, four issue
  comments, one review and zero inline comments; one commit.
- Reviewed files: runtime `vllm/config/compilation.py` and
  `vllm/config/vllm.py`; `tests/compile/test_config.py`; `docs/design/fusions.md`.
- Motivation: the default split removes the very cache-update operation an
  explicitly requested fusion must rewrite. Keeping it inside the graph is
  practical when layer names are graph inputs rather than per-layer constants.

The patch retains the ordinary cache op when either ordinary or QK fusion
requests it and hoisting is enabled. It independently handles the MLA cache
op, preserving unrelated cache splits and attention splits. Without hoisting,
the helper warns and refuses the requested default-splitting-path fusion.

```diff
+                    if not self._keep_kv_cache_update_in_graph(
+                        self.pass_config.fuse_rope_kvcache_cat_mla,
+                        "MLA RoPE+KV cache fusion",
+                    ):
+                        self.pass_config.fuse_rope_kvcache_cat_mla = False
+                        self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
```

It moves compile-range resolution after splitting decisions and gives the
QK resolver the same graph-visibility check as ordinary RoPE/cache. It does
not resolve the optimization-level/custom-op ordering cycle; #57900 adds that
part. Both patches affect shared configuration across model families, not
only Llama. They change reachability of existing kernels rather than their
arithmetic or launch implementation.

The six added parameterized tests exercise three requested flags with and
without hoisting, including preservation of the other cache op and attention
split. Tests set flags after platform validation to make this configuration
logic testable without the corresponding GPU. The hoisted lane is skipped
when `_USE_LAYERNAME` is false. The body reports eight splitting tests passing
on MI325X/PyTorch 2.12 after rebasing to `7128f568ac`; these are not GPU
correctness tests of every fused backend.

Historical MI355X measurements predate that rebase (`0aee727ff6`) and were
explicitly not rerun. The reported compile cost is +0.08 seconds per range
with hoisting versus 1.26→16.02 seconds without it. The older kernel claim of
48–51% improvement over 1–64 tokens and break-even around 192–256 is superseded
by #57900's more detailed shape table, which includes regressions.

Older serving uses random 128/1024 lengths, 600 prompts, concurrency 50 and
three runs per arm. Llama median TPOT/TTFT improve 2.6%/14.9%; only those two
reported intervals are disjoint. Mixtral TPOT improves 4.1% but TTFT regresses
3.0%; Qwen3 matches zero patterns and its small deltas are not fusion evidence.
The 1,319-question, five-shot greedy GSM8K results do not establish identical
outputs: Mixtral text agreement is 55.5%, versus 99.6% between baseline repeats.
Reported ordinary-pass matches on Llama/Mixtral/gpt-oss do not validate QK-norm
or MLA kernels. Discussion contains conflict/automation messages; a bot saying
review is disabled is not a successful validation result.

## PR #57900 — Resolve both explicit requests and optimization-level opt-ins

- Link: https://github.com/vllm-project/vllm/pull/57900
- State: open; head `65551c01c2489b3c105ae45d9c0f6fe785c2dda6`, base
  `61968a50d578ba9078dcf036eaf16d688d10cc46`. It supersedes #57349 and is not
  present in the frozen main configuration above.
- Coverage: all 347 diff lines, five files, +202/-36; body, five issue
  comments, one review and zero inline comments; two commits.
- Reviewed files: the four files from #57349 plus `tests/test_config.py`.
- Motivation: fixing the split is insufficient when optimization defaults
  wait for a custom op that is only requested after the fusion flag is enabled.

Alongside #57349's hoisting/splitting change, the default resolver now treats
an explicit `-rotary_embedding` as an opt-out instead of requiring the custom
op to be enabled beforehand:

```diff
     return (
         rocm_aiter_ops.is_enabled()
-        and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
+        and "-rotary_embedding" not in cfg.compilation_config.custom_ops
         and (
             cfg.compilation_config.use_inductor_graph_partition
             or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
```

The QK resolver delegates to this same predicate. After splitting settles,
the code appends `+rotary_embedding` if a surviving ordinary/QK/standalone
QK-RoPE flag needs it, avoiding a duplicate positive entry, then computes
compile ranges. The explicit-negative test covers default resolution; it
does not prove that all contradictory explicit flag/custom-op combinations
are rejected. The two cache-op families still have separate retention decisions.

New configuration tests monkeypatch AITER availability and check O2 with
Inductor partition or an empty splitting list, stock defaults, custom-op
opt-out and AITER disabled. The author reports splitting selection 6 failed /
3 passed→9 passed and resolver selection 2 failed / 3 passed→5 passed. The
20-checkpoint stock O0–O3 configuration comparison is not a model-output test;
DSV4.1 differs in an unused flag while running without `torch.compile`.
Stock optimization levels alone still retain the default split and do not
enable these fusions. Explicit requests and the documented partition/empty-list
opt-ins are the changed cases.

### Validation and measured tradeoffs

Tests/config checks were rerun on base `61968a50d`; operator and serving
measurements used earlier base `bc21bd921`. The author reports unchanged
patch logic across the intervening 201 commits, apart from import context;
that is a provenance qualification, not a fresh benchmark of frozen main.

The GPU setup is MI355X/gfx950, Llama-3.1-8B BF16, TP1, `ROCM_ATTN`, AITER
0.1.27 and PyTorch 2.13. Both arms explicitly request the ordinary fusion;
the baseline drops it but retains custom rotary, so the comparison does not
mix native rotary with custom rotary. Operator inputs are views of one QKV
buffer, 32 Q / 8 KV heads, head size 128 and cache block size 16. A HIP graph
contains 200 calls; the table reports the median of seven replays.

| Tokens | Unfused, µs | Fused, µs | Latency change |
|---:|---:|---:|---:|
| 1 | 4.99 | 2.41 | −51.8% |
| 16 | 5.39 | 2.70 | −49.9% |
| 64 | 5.85 | 4.13 | −29.4% |
| 128 | 6.54 | 7.21 | +10.3% |
| 256 | 8.17 | 8.54 | +4.5% |
| 512 | 12.36 | 10.74 | −13.1% |

The default 256-token threshold is unchanged despite the 128/256 regressions.
512 is outside the fused compile range and therefore is not a serving gain.
About 21% of Q/K elements differ by 1–2 BF16 ULP; K cache follows K, while V
cache is identical in this report. Above the fused range the forced custom
rotary op can still affect performance; evaluate the full serving shape mix.

Serving uses max length 4096, no prefix cache, memory utilization 0.9,
random 128/512 lengths with range ratio 0.25, ignore-EOS and seed 1234;
32 prompts per concurrency slot. Four fresh-server runs per arm are ordered
baseline first, with the first discarded and the median of three retained.
At concurrency 32/64, total throughput improves 2.8%/2.2% and median TPOT
4.74→4.61 / 5.62→5.51 ms. Reported ranges do not overlap, but measured kernel
savings account for only about half the TPOT delta; the remainder was not
isolated. These are finite-burst measurements of one model/backend/shape setup.

Full 1,319-question, five-shot greedy GSM8K strict/flexible scores are baseline
0.6975/0.7794, repeat 0.6945/0.7741, patch 0.6975/0.7733. Flexible-score loss
0.0061 is slightly larger than the 0.0053 baseline-repeat change; retain those
numbers rather than certifying the body's broad claim of staying within drift.
Per-sample verdict agreement is 93.5%, versus 96.8% for the baseline repeat.
Neither configuration equality nor these task scores establishes equivalence.

One ordinary pattern per piecewise artifact can serve all 32 Llama layers
through hoisting; the partitioned graph reports 32 matches. Qwen3 needs the
combined QK-norm pattern rather than the ordinary split→RoPE chain. The author
reports 36 combined matches on Qwen3-8B and MLA matches on Kimi-K2.7, but neither
variant was benchmarked here. The PR changes config/docs/tests only, so those
match counts do not validate every backend or layout admitted by current main.
The discussion's conflict notifications and disabled automated review provide
no additional final-head GPU result.

## PR #49142 — Default MLA fusion without splitting its cache update

- Link: https://github.com/vllm-project/vllm/pull/49142
- State: open; head `8c1e07b12abb252d49328c85c2312559a14997f6`, base
  `36f94d5fe8a3953f1ad340de3c5aaec909765243`; fresh head/base still match.
- Coverage: all 59 diff lines, three files, +20/-6; body, seven issue comments,
  five reviews and three inline comments; nine commits.
- Reviewed files: `vllm/config/compilation.py`, `vllm/config/vllm.py` and
  `tests/compile/test_config.py`. No kernel or serving harness is added.
- Motivation: an existing MLA RoPE/concat/cache replacement cannot match if
  Dynamo removes the cache-update op before the pattern pass sees it.

The MLA default resolver changes from graph-visibility conditions to
`current_platform.is_cuda_alike()`. In the default split builder, an enabled
MLA flag keeps only its own cache-update op in the graph:

```diff
                     self.splitting_ops.append("vllm::unified_kv_cache_update")
-                    self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
+                    if not self.pass_config.fuse_rope_kvcache_cat_mla:
+                        self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
```

The new assertion verifies that regular cache updates remain split while
MLA updates do not. This is a platform-only default resolver, unlike #57900's
explicit-request retention guarded by `_USE_LAYERNAME`; this patch has no
hoisting check. It affects MLA families sharing the compiler config, including
CUDA-like platforms beyond the H100 used for the reported experiment. Current
frozen main still uses the older visibility resolver and unconditionally adds
the MLA cache split in this branch.

Review explicitly asked why partitioning was no longer needed. The author
explained that keeping the op visible replaces that prerequisite, then
reported DeepSeek-V2-Lite dummy-weight startup 22.501→22.717 seconds with
identical AOT/Inductor counters (zero misses, six hits each). Those counters
do not demonstrate uncached compilation or absence of per-layer specialization
on configurations without hoisted layer names. An approval and final-head CI
trigger are recorded. The author attributes earlier CUDA-graph/startup failures
to NVML permissions and missing `nvrtc.h`; these comments do not establish a
clean final-head full CI run.

The body reports one splitting test, four Triton MLA cases and six supported
fusion cases passing, excluding FlashAttention MLA with FP8 cache. Current
`test_mla_rope_kvcache_cat_fusion.py` was read separately: it uses a five-row
synthetic model, DeepSeek-scaled RoPE, NeoX/interleaved forms, BF16 and auto/FP8
cache, with available backends. It explicitly initializes dense projection
weights and compares intermediate outputs/cache after one compiled call;
`FixFunctionalizationPass` preserves noncontiguous slice offsets. This is
useful operator coverage, not multi-layer serving, repeated CUDA Graph replay,
HiSparse or task-accuracy validation. Current fused and unfused cache wrappers
both prepare the layer's cache state, but backend-specific write ownership
still needs checking before broadening defaults.

Historical serving covers 100 requests on one H100 with DeepSeek-V2-Lite,
6,400 output / 7,100 total tokens. Reported output throughput is 5,959.83 tok/s
baseline versus 10,596.56 and 11,135.59 in two candidate runs; mean TPOT is
0.168 versus 0.094/0.090 ms. The named `bench_mla_serving.py` is not in this
diff, and concurrency, arrival rate, warmup, precise tested SHA and repeated
baseline measurements are not specified. Preserve these as historical reports,
not a general MLA speedup or confirmation of model accuracy.

## PR #54851 — Bind real DSA caches during metadata-free capture

- Link: https://github.com/vllm-project/vllm/pull/54851
- State: open; head `7d99c3d6881f97f057148ac85313fc8ab241c888`, base
  `385dce36bcee42309924a5ece951a96db3dce7f2`; fresh head/base still match.
- Coverage: all 49 diff lines, one file, +26/-1; body, two issue comments,
  one review and zero inline comments; one commit.
- Reviewed file: `vllm/models/deepseek_v32/attention.py`. No kernel change,
  automated regression test or benchmark harness is included.
- Motivation: if attention Python executes with metadata absent during graph
  capture, passing absent caches/slots into the captured fused writer can
  permanently disable its cache writes on replay.

The patch separates PCP from the metadata-free case. PCP still materializes
KV outside the fused cache writer. Without metadata, an allocated MLA cache
and non-null persistent slot buffer retain the real cache and scale pointers;
the indexer cache is retained only when also present and allocated:

```diff
+            if mla_slot is not None and self.kv_cache.numel() > 0:
+                mla_kv_cache = self.kv_cache
+                mla_k_scale = self._k_scale
+            else:
+                mla_kv_cache = None
+                mla_k_scale = None
+                mla_slot = None
```

The intended lifetime contract is a stable address with changing contents:
capture uses padded negative slots, and runtime updates that same buffer with
real slots before replay. A newly allocated replacement slot tensor would not
repair the address already recorded in the graph. When caches are absent, the
wrapper instead uses an empty MLA tensor/zero entry stride and no cache slots;
replay of that launch cannot discover newly bound Python tensors on its own.

### Current caller and capture qualifications

Frozen main still has the `attn_metadata is None or self.use_pcp` branch that
clears both cache paths and slot mappings. However, current MRV2 capture
explicitly builds attention metadata for both FULL and PIECEWISE, using
`for_capture=False` for PIECEWISE, and passes the result into forward context.
The default model-state implementation calls the metadata builder even when
that flag is false. Therefore `for_capture=False` alone does not establish
`attn_metadata is None` or reproduce the reported corruption on current main.
Inspect the actual runtime context/runner and captured launch before applying
this diagnosis to a new trace. The body describes the older failing setup;
the source review establishes a conditional mechanism, not a current GPU repro.

Current `get_dummy_slot_mappings` fills the persistent allocation with
`PAD_SLOT_ID`; normal slot computation reuses it. Capture also supports a
custom persistent mapping buffer and pads it before recording. In the current
fused kernel, Q normalization/top-k clearing can still execute on padded rows,
and materialized KV outputs may be needed for dense prefill or PCP. Negative
slots protect cache writes; they do not make the entire launch a no-op.

Current attention also has independent MLA and indexer mappings, shuffled
indexer cache layouts, HiSparse's separate cache-update ownership and PCP
gather/write paths. The old patch only handles its earlier single-slot branch;
a port must preserve these newer contracts instead of blindly rebinding every
cache. The compiler MLA default change #49142 and this handwritten DSA
prologue fix operate on different paths and are not interchangeable fixes.

### Validation limits

The author reports a four-node AGX Thor/SM110 TP4 setup using GLM-5.2
NVFP4-AQLM-hybrid, `TRITON_MLA_SPARSE`, `fp8_ds_mla` and PIECEWISE graphs.
Baseline `80389cfedd` produced repeated garbage; the patch restored coherent
greedy short, medium and 2,000-token responses. Device probes reportedly
show finite indexer/MLA values and context-growing valid top-k counts. This is
manual smoke/diagnostic evidence, not GSM8K/AIME accuracy or a checked-in
capture/replay regression. The approximate 10× decode claim compares against
a corrupt ~0.1 tok/s run and is not a normal kernel-performance baseline.

The separate standalone harness's out-of-bounds-slot failure is explicitly
not the reported server trigger; this diff adds no bounds check. Similar
symptoms in other models, DCP/MTP or FULL_AND_PIECEWISE do not establish the
same cause. The discussion contains a conflict notice and disabled automated
review, with no human approval or additional final-head test results.

Current source:
[configuration and splitting](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/compilation.py),
[default resolution and ranges](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/vllm.py),
[hoisted layer names](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/utils/torch_utils.py),
[pass manager](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/pass_manager.py),
[ordinary RoPE/cache](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/rope_kvcache_fusion.py),
[QK norm/RoPE/MRoPE/cache](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py),
[MLA RoPE/concat/cache](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py),
[ROCM_ATTN layout and fallback](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/attention/backends/rocm_attn.py),
[AITER wrappers](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/_aiter_ops.py),
[current splitting tests](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/tests/compile/test_config.py),
[MLA fusion numerical tests](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py),
[MLA cache preparation and update](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/layers/attention/mla_attention.py),
[DSV3.2 attention caller](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/models/deepseek_v32/attention.py),
[fused norm/RoPE/cache kernel](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/models/deepseek_v32/common/kernels.py),
[MRV2 graph capture](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/worker/gpu/cudagraph_utils.py),
[persistent slot buffers](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/worker/gpu/block_table.py),
[default attention metadata preparation](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/worker/gpu/model_states/default.py).
