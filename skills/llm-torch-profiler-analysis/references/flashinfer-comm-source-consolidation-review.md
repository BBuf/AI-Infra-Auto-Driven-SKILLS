# FlashInfer communication source consolidation

Reviewed on 2026-11-09 (Asia/Shanghai). Historical source is pinned below;
current comparisons use `31c146b16789f112e9b9861bd79247d4701e437a`. This is
a source-maintenance review, with local CPU equivalence checks. No CUDA
compilation, SASS comparison, GPU capture or performance experiment was rerun.

## PR #7054 — one source per clone group, one binding, one prelude header

- Link: https://github.com/flashinfer-ai/flashinfer/pull/7054
- State: merged on **2026-10-23**, head
  `b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f`, base/merge-base
  `0602a5b9929d0bc2322d16c56088d74b6e0413aa`, merge
  `39f9462f7d7f07868df36175a5c3e556355e30a0`.
- Coverage: all 45 changed files (+394/-9,771; 11,234 diff lines), body,
  two issue comments, two reviews, one inline comment and five commit
  messages. Unique edits were read; removed clones and repeated generated
  edits were checked over their complete contents by byte comparison or
  preprocessing as detailed below. This does not independently re-audit the
  unchanged 6,533-line all-to-all algorithm or every unchanged Ulysses body.
- Reviewed paths: `csrc/generated/moe_alltoall/`, the union kernel directory,
  norm-combine bindings, the finalize binding comment,
  `flashinfer/cake_ulysses_backend/{build.json,csrc/}`, the four changed JIT
  loaders and both changed communication tests. Supporting reads include the
  Ulysses manifest loader, JIT build/cache path and Ninja dependency rules.
- Motivation: remove duplicated generated source while preserving separate
  programs, compile settings and runtime dispatch. A shared source file does
  not make two launch-bound or pipeline variants interchangeable.

### What actually becomes shared

| Family | Change at this PR head | What remains distinct |
| --- | --- | --- |
| MNNVL MoE all-to-all | Two byte-identical 6,533-line CUDA files become `cake_moe_alltoall_kernels.cu`. | SM100a/SM103a module names and architecture flags. |
| Dense MoE reduction union | Seven programs use three retained pipe sources; four duplicate files are deleted. Twenty other device sources are byte-identical. | All 27 program names, route entries, symbols, dtype/world-size and launch geometry. |
| Indexed MoE finalize | Remove unused loader tables and `kernel_names()`; add a generated-source comment to its binding. | Runtime kernel and dispatch; `HIDDEN_DIM` is retained for its benchmark caller. |
| Eight-peer norm-combine | Three host bindings become one binding with five required compile definitions. | Three device files, shim namespaces, block/shared-memory sizes, cooperative attributes and occupancy helper. |
| Ulysses all-to-all | Move the common prelude of 24 kernel files into one header; update 48 manifest source digests. | Forty-eight library names and closure digest values, corresponding to two architectures. |

The union loader's `Kernel` gains optional `source` and `defines` fields.
The kept BF16 TP2 program defaults to pipeline depth two and a minimum of
five blocks per SM. Its siblings select depth one or omit the minimum-block
launch bound. FP16 TP2 selects depth one/two; BF16 TP8 additionally selects
expert-loop unroll four/eight. The source uses literal preprocessor arms,
including the pipeline drain and consumer offset:

```diff
+#if CAKE_UNION_PIPE_DEPTH == 2
         if (step >= 2) {
             int consume_token = cluster_id + (unsigned int)(step - 2) * num_clusters;
+#else
+        if (step >= 1) {
+            int consume_token = cluster_id + (unsigned int)(step - 1) * num_clusters;
+#endif
```

The [historical loader](https://github.com/flashinfer-ai/flashinfer/blob/b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f/flashinfer/jit/cake_trtllm_moe_allreduce_union.py)
passes `compile_flags(name, arch)` to **both** the device source and shared
launcher. Thus the body's generic assertion that only the shared source
receives the new definitions needs qualification: the launcher sees them too.
Local preprocessing verifies that the four folded records' additional
definitions are inert in that unchanged launcher. Norm-combine instead uses
`extra_cuda_cflags_by_source`: its five binding definitions are scoped to the
binding, while the cooperative occupancy source gets only its own kernel
definition and common flags.

The [union device arms](https://github.com/flashinfer-ai/flashinfer/blob/b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f/csrc/cake_trtllm_moe_allreduce_union/kernels/ws2_bf16_pipe2_u4_b5.cu)
retain expert-major addressing, FP32 route-scale multiplication followed by
activation-dtype rounding, intermediate rounded additions, positive-zero
canonicalization, Lamport sentinel polling and the alternating DSM partial
slots. FP16 and BF16 still have their own conversion/addition code; this PR
does not replace either with a generic FP32 reduction. The dense reduction
contract remains distinct from the inverse-index-map
[indexed finalize API](flashinfer-moe-finalize-source-review.md).

The norm binding preserves all predicates and error messages, but error
tracebacks now identify the shared file and its line numbers. Its launch
error string uses adjacent literals after macro stringization. Treat this
diagnostic-location change separately from runtime equivalence. Removed
`exported_arches()` and finalize names were exported Python loader symbols:
an absence of in-tree users does not prove compatibility for external direct
imports. The per-rank tuple branch of `Route.kernel` had no route and is
removed; rank-bound checks remain.

### Header integrity is separate from JIT freshness

The inline review flags that the new Ulysses header is absent from the
manifest's `sources` and `sha256` map. The unchanged
[`_verified_sources`](https://github.com/flashinfer-ai/flashinfer/blob/b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f/flashinfer/cake_ulysses_backend/cake_build.py)
checks listed sources only. A local CPU reproduction executed that original
function against an actual entry and copied source files: appending an
`#error` to the included header still passed verification; mutating a listed
kernel source raised the expected digest error. The same experiment at the
current frozen main revision has the same result. This proves a manifest
integrity gap for included headers, not stale execution or GPU corruption.

The build path provides a separate mechanism:

- [`JitSpecNvcc.try_load()`](https://github.com/flashinfer-ai/flashinfer/blob/b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f/flashinfer/jit/core.py)
  accepts known AOT artifacts; the JIT fallback runs the build, writes the
  Ninja file if changed and calls Ninja even when a JIT library exists.
- [The CUDA compile rule](https://github.com/flashinfer-ai/flashinfer/blob/b2fae49ab2d264bed7ed458fc5ba5c344dc40f7f/flashinfer/jit/cpp_ext.py)
  uses `--generate-dependencies-with-compile -MF $out.d`, `depfile` and
  `deps = gcc`. This lets Ninja track included headers in a normal build.
- An unchanged `closure_sha256` or module name therefore does not by itself
  demonstrate stale JIT reuse. Conversely, dependency timestamps are not a
  cryptographic source-integrity check, and an already loaded cached module
  is not a promise of hot reload. No local rebuild experiment was performed.

The inline concern has no substantive human resolution in the reviewed
thread; the approval body is empty. Do not describe the manifest gap as
fixed. The standing manifest checker and `cake_build.py` were unchanged.

### Validation: source equivalence, reported GPU gates and actual CI

Local CPU checks used Clang preprocessing with CUDA/FFI includes consistently
removed, then compared tokens. They cover all seven folded union programs at
`__CUDA_ARCH__=1000/1030` (14 comparisons), all three norm bindings under their
actual definitions, and all 24 Ulysses sources with the header expanded at both
architectures (48 comparisons). Adjacent string literals were joined for the
norm binding comparison. The untouched 20 union and three norm device files
were byte-identical. Both old all-to-all files and the retained file have
SHA256 `c634bc0cd4671a942df36dadb2023f8b8e0360a23b9e2e4101239727d95fd571`.
Parsing both Ulysses manifests shows exactly 48 kernel-source digest changes
and no other structural change; all listed new source digests match their
files. These checks do not expand vendor headers, compile host objects or
establish SASS/resource equivalence.

The PR author reports 68 SASS rows (14 union, six norm-combine, 48 Ulysses),
two additional MNNVL rows, 62 device-preprocessing rows, eight host rows and
eight launcher-inertness rows. Each architecture builds 56 modules per tree
through the real loaders. The two informational comparisons between
`__launch_bounds__(224)` and `(224, 5)` have **different masked SASS**; neither
variant is deleted. Do not collapse their dispatch choices based on source
similarity. The reported Blackwell test outcomes on both trees are:

| Node | MNNVL | Union/distributed | Norm-combine | Ulysses | Finalize |
| --- | --- | --- | --- | --- | --- |
| B200, eight GPUs | 203/0 | 165/1 | 23/0 | 18/0 | 11/0 |
| GB300, four GPUs | 203/0 | 163/3 | 22/1 | 14/4 | 9/2 |
| B300, eight GPUs | 203/0 | 165/1 | 23/0 | 18/0 | 11/0 |

Cells are **passed/skipped**, quoted author reports, not this refresh's
measurements. The author compares per-test outcomes and skip reasons;
the tests do not save tensors for a cross-tree output comparison. All harnesses
are single-node. The body explicitly says **no benchmarks were run**, despite
its generic opening performance claim. There is no measured speedup from
this cleanup to add to a profiler recommendation.

Independently inspected final-head
[H100 CI](https://github.com/flashinfer-ai/flashinfer/actions/runs/39865712422/job/120127480030)
checks out merge `690add8` of this exact head into `0602a5b9`. Its nine-file
selection finishes **215 passed, 196 skipped**, with no failures or unknown
results. The union file contributes 154 passes; all six distributed reduction
cases skip, as do all 163 all-to-all cases and 17 multi-GPU all-to-all cases.
Norm-combine has 22 passes/one skip, Ulysses ten/eight, the fused adapter
18/one, API six/zero and finalize dispatch five/zero. These results and the
six successful AOT build/import jobs do not validate Blackwell communication
execution on H100. The test's new source-symbol regex verifies the literal or
macro spelling; it is not a substitute for compiling every folded record.

### Current source and profiler use

At the frozen current main revision, the shared MNNVL file is still
byte-identical to this PR's retained file. The union portfolio has later
changes; the [collective review](flashinfer-moe-collective-source-review.md)
records current selection separately. Do not use this PR's 27-program count
as a current registry count. Indexed finalize also has a later generated
portfolio rather than the historical single template.

The norm-combine shared binding from this PR has since been deleted. The
[current registry](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/jit/cake_fused_norm_combine.py)
contains six `CakeProgram` records with separate generated bindings. The
[current host selector](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/comm/cake_fused_norm_combine.py)
uses one-token specialization, ordinary parallel for T=2–3, split parallel
for T=4–37, owner streaming for T=38–479, and pipelined selection for T≥480.
Within the last band, the early-load variant requires
`4 * SM_count < T <= 4 * persistent_grid(T, 3 * SM_count)`; otherwise it uses
the regular pipeline. Persistent grids use all tokens when they fit, else
the largest odd grid within capacity. Three persistent programs are
cooperative. This replaces the old three-band/1024-token summary; these host
rules are not a fresh audit of all six device bodies or their originating PRs.

The current Ulysses manifest has 96 libraries, and its shared prelude now
includes another helper header. Its unchanged source verifier still has the
gap reproduced above; the historical 48-library identity proof does not
cover later exports. When attributing a trace, record the deployed program,
source revision, target, compile definitions and actual dispatch. Preserve
separate variants with different launch bounds, residency or pipeline depth,
even when they share a source path. Use timing and correctness evidence from
that variant; source consolidation alone supplies neither.
