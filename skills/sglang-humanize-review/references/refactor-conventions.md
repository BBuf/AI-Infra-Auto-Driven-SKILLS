# SGLang layout and cleanup conventions from BBuf's refactor PRs (2026-06 to 2026-10)

Rules from merged refactor and cleanup PRs in `sgl-project/sglang`, plus the guard tests and lints they added, for reviewers to apply to incoming diffs. Source of truth on `main`:

- `python/sglang/kernels/README.md`
- `python/sglang/kernels/ops/diffusion/README.md`
- `test/README.md`
- the `kernel-organization` skill

Each rule gives **Why**, **Detect** (what to look for in the diff) and **Src** (public PRs).

## 1. Kernels (`python/sglang/kernels`)

**KR1. Import callable kernels from `sglang.kernels.ops.<group>`, and add new kernels there.**
- Why: RFC [#29630](https://github.com/sgl-project/sglang/issues/29630) made `sglang.kernels` the single home for kernels. All `srt/**/triton_ops` trees and about 280 scattered kernels were moved in, and `sglang.jit_kernel` was deleted.
- Detect:
  - `from sglang.jit_kernel`, or a new `triton_ops/` directory.
  - A new `@triton.jit`, CuTe-DSL or `load_jit(...)` kernel defined under `srt/` or `multimodal_gen/`.
  - Runtime code calling `sgl_kernel` directly where an `ops.<group>` wrapper exists.
  - Exceptions: the `sgl_kernel` wheel API and vendored trees.
- Src: [#30044](https://github.com/sgl-project/sglang/pull/30044), [#30784](https://github.com/sgl-project/sglang/pull/30784)–[#30795](https://github.com/sgl-project/sglang/pull/30795), [#31582](https://github.com/sgl-project/sglang/pull/31582), [#32072](https://github.com/sgl-project/sglang/pull/32072), [#26000](https://github.com/sgl-project/sglang/pull/26000).

**KR2. Group an operator by its complete computation, not by the model that first used it.** The rules:
- Any cross-rank collective or peer-memory access goes in `communication`.
- A producer plus its output quantization stays with the producer: norm+quant in `layernorm`, act+quant in `activation`, a quant-epilogue GEMM in `gemm`, MoE dispatch+quant in `moe`.
- `quantization` holds only pure format conversion.
- RoPE and QK-norm+RoPE go in `attention`.
- `kvcache` owns ops that read or write cache contents. `attention` owns page tables, `cu_seqlens` and `kv_indices`.
- A GEMM or GEMV and its tuning JSON go in `gemm`.

Further notes:
- Why: model-named bundles and misfiled ops kept drifting back in.
- Detect:
  - A new `kernels/ops/<model>/` directory, or a `.py` placed directly in `ops/`.
  - A quantized GEMM under `quantization/`, or an all-reduce-fused kernel outside `communication/`.
  - Do not flag a model-named *file* inside the right group.
- Src: [#45112](https://github.com/sgl-project/sglang/pull/45112), [#41243](https://github.com/sgl-project/sglang/pull/41243), [#32148](https://github.com/sgl-project/sglang/pull/32148).

**KR3. A new group or top-level `kernels/` directory updates `ops._GROUPS`, the README group list and the README Layout block in the same PR.**
- Why: `test/registered/unit/kernels/test_kernels_namespace.py` fails otherwise:
  - `test_readme_operator_groups_match_registry`
  - `test_readme_layout_lists_every_top_level_directory`
  - `test_operator_and_test_groups_agree`: no `.py` directly in `ops/`, and every test or benchmark subdirectory must be a real group.
- Detect: a new directory with no README or `_GROUPS` change.
- Src: [#45112](https://github.com/sgl-project/sglang/pull/45112), [#32788](https://github.com/sgl-project/sglang/pull/32788).

**KR4. Register every public entry point with a lazy `KernelSpec(op="<group>.<name>", backend=..., target="module:attr")`, and keep group imports metadata-only.**

Why: guard tests enforce this, and the registry is checked too:
- `test_import_stays_metadata_only` fails if importing `sglang.kernels.ops` loads `sgl_kernel`, `sglang.kernels.jit`, cutlass, flydsl or aiter.
- `test_inventory_targets_survive_module_moves` fails on stale targets.
- `test_reclassified_public_entry_points_are_inventoried` requires specs for public functions. Names starting with `_`, `can_`, `is_` or `use_` are exempt.
- The registry rejects conflicting registrations for the same op and backend.

Detect:
- A new public function in `kernels/ops/**` with no spec, or heavy imports in a group `__init__.py`.
- Write-only metadata (`description=`, `FormatSignature`), which #44690 deleted.
- Inline CUDA-only frozensets instead of `spec.CUDA_ONLY`.

Src: [#30044](https://github.com/sgl-project/sglang/pull/30044), [#32636](https://github.com/sgl-project/sglang/pull/32636), [#32788](https://github.com/sgl-project/sglang/pull/32788), [#44690](https://github.com/sgl-project/sglang/pull/44690).

**KR5. `KernelBackend` names provenance (triton, jit, aot, cute_dsl, flashinfer, …), never a device. Devices go in `CapabilityRequirement`.**
- Agent and plugin directories pair with their backend in both directions (`test_backend_directories_register_their_backend`):
  - `kda_kernels/` ↔ `KDA`
  - `cake_kernels/` ↔ `CAKE`
  - `plugin_kernels/` ↔ `PLUGIN`
- Runtime code imports the `ops.<group>` facade, never these packages.
- Why: one provenance runs on several devices, and mislabelled agent kernels were missing from a filter by backend.
- Detect:
  - Enum values such as `CUDA_JIT` or `HIP_TRITON`.
  - A spec targeting `kda_kernels.` with a language backend.
  - `srt/` importing those packages.
  - A PR that reorganizes them.
- Src: [#31292](https://github.com/sgl-project/sglang/pull/31292), [#31307](https://github.com/sgl-project/sglang/pull/31307), [#37385](https://github.com/sgl-project/sglang/pull/37385), [#45277](https://github.com/sgl-project/sglang/pull/45277).

**KR6. Keep runtime state in `srt/` and pass it to kernels as arguments. Kernels never import `srt`.**
- Runtime state means process groups, symmetric buffers, workspaces, communicator registries, config parsing and model dispatch. It lives in `srt/layers/communication/`, `srt/distributed/` or `srt/kernel_plugins/`.
- Why: reverse imports pin the layout and duplicate singleton buffer registries.
- Detect: `from sglang.srt` under `python/sglang/kernels/`, or a kernel module that defines a workspace class or inspects model objects.
- Src: [#45112](https://github.com/sgl-project/sglang/pull/45112), [#31582](https://github.com/sgl-project/sglang/pull/31582).

**KR7. Split external whole-step kernel plugins at the kernel/runtime boundary, and name them by hardware.**
- The contract goes in `kernels/plugin_kernels/<hardware>/<model>/`, uses `KernelBackend.PLUGIN`, and imports nothing from `srt`.
- SGLang's integration goes in `srt/kernel_plugins/<hardware>/binding.py`.
- The switches are boolean flags: `--enable-nvidia-megakernel` and `--enable-amd-monokernel`.
- An old spelling stays only as a deprecated alias that warns and forwards, and that still works in `--config` YAML.
- Detect: new `srt/megakernels`-style packages, value-typed `--xxx-backend <vendor>` flags, or a deprecated alias with no YAML regression test.
- Src: [#45389](https://github.com/sgl-project/sglang/pull/45389).

**KR8. Move code with `git mv` plus import rewrites, keep relocation commits separate from semantic ones, and leave no forwarding shim.**
- Update every reference in the same PR:
  - dotted and package-relative imports, `__init__` re-exports and `mock.patch` strings
  - lazy spec targets and tuning-file paths
  - benchmarks, docs, CODEOWNERS and CI path filters
- Why: missed import forms broke main twice: a test's second import site in #45112, and package-relative imports before #32072.
- Detect: a module that only re-exports moved symbols, an old path that `git grep` still finds, or tuning JSON moved without its loader path.
- Src: [#32015](https://github.com/sgl-project/sglang/pull/32015), [#32072](https://github.com/sgl-project/sglang/pull/32072), [#41243](https://github.com/sgl-project/sglang/pull/41243), [#45112](https://github.com/sgl-project/sglang/pull/45112).

**KR9. Keep test, benchmark and debug code out of operator modules.**
- Reference oracles go in `sglang.test.kernels`, and standalone scripts go in `test/manual/kernels/`.
- JIT wrappers live at `ops/<group>/<op>.py` with no `_jit_` prefix. CUDA sources live in `kernels/jit/csrc/<area>/`, shared build infrastructure in `kernels/jit/`, and the AOT wheel source in `kernels/aot/`.
- Detect: reference helpers or `if __name__ == "__main__"` in `kernels/ops/**`, or a loose header placed directly in `jit/`.
- Src: [#32128](https://github.com/sgl-project/sglang/pull/32128), [#32148](https://github.com/sgl-project/sglang/pull/32148), [#32648](https://github.com/sgl-project/sglang/pull/32648), [#36704](https://github.com/sgl-project/sglang/pull/36704), [#44690](https://github.com/sgl-project/sglang/pull/44690).

**KR10. Delete unreachable kernels, reuse shared device helpers, and keep a duplicate implementation only when it is not bit-exact with its twin, with a one-line comment saying so.**
- "Unreachable" includes kernels that only tests call.
- Shared helpers include `AlignedVector`, warp/bits/PDL utilities, TMA descriptor builders and CuTe argument converters.
- Detect: a hand-rolled `union` vector type, copied TMA or argument helpers, or a new kernel with no runtime caller.
- Src: [#29664](https://github.com/sgl-project/sglang/pull/29664), [#32636](https://github.com/sgl-project/sglang/pull/32636), [#32842](https://github.com/sgl-project/sglang/pull/32842), [#40197](https://github.com/sgl-project/sglang/pull/40197), [#44690](https://github.com/sgl-project/sglang/pull/44690).

## 2. Fused ops (`BaseFusedOp`)

**F1. A multi-backend operator subclasses `sglang.kernels.fused_op.BaseFusedOp` and implements a real pure-torch `forward_native`.**
- Why: `forward_native` is the reference for the parity harness and for the `SGLANG_FORCE_FUSED_OP_BACKEND=torch` bisect switch. `MultiPlatformOp` is now a deprecated alias.
- Detect:
  - `class X(MultiPlatformOp)`.
  - A `forward_native` that raises (#45112 fixed one).
  - An unreviewed `forward_native = forward_cpu` alias.
  - Compile special-cases keyed on class names instead of overriding `_torch_compile_forward()`.
- Src: [#33205](https://github.com/sgl-project/sglang/pull/33205), [#45112](https://github.com/sgl-project/sglang/pull/45112).

**F2. Keep kernel backends and platform paths separate.**
- Kernel backends are `forward_<backend>` methods (`forward_triton`, `forward_jit`, `forward_torch_npu`, …).
- Platform paths are `forward_cuda`, `forward_hip`, `forward_npu`, `forward_xpu`, `forward_musa` and `forward_cpu`.
- A backend joins auto-selection only when it is overridden **and** declared in `capabilities`.
- Only HIP falls back to CUDA implicitly. MUSA must opt in with an explicit `forward_musa`.
- Detect:
  - A `forward()` override with device `if/else` chains.
  - A torch_npu backend named `forward_npu`.
  - A backend missing from `capabilities`.
  - A new enum value with no `forward_<name>` stub (`FLYDSL` raised `AttributeError` this way).
- Src: [#33205](https://github.com/sgl-project/sglang/pull/33205), [#44690](https://github.com/sgl-project/sglang/pull/44690).

**F3. Gates answer "is this kernel applicable"; launchers enforce the operand contract.**
- A `can_use_*` gate or `backend_eligible()` checks dtype, arch, layout, alignment and real specializations.
- Mismatched shapes or devices between operands raise in the launcher: `TensorMatcher`/`CHECK_HOST` in C++, or the Python wrapper for Triton.
- Why: a gate that restates the contract turns a caller bug into a silent, slower fallback, and it bypasses the launcher check.
- Detect:
  - A gate that takes tensors and compares their shapes.
  - Flattening before validation.
  - `except Exception:` that sets a process-wide `_DISABLED`.
  - A kernel that returns `None` for "unsupported"; it must raise, with a public predicate.
  - Choosing CPU vs CUDA from anything other than the input tensor's device.
- Src: [#42391](https://github.com/sgl-project/sglang/pull/42391), [#44690](https://github.com/sgl-project/sglang/pull/44690), [#35114](https://github.com/sgl-project/sglang/pull/35114).

## 3. MoE

**M1. Route gate and top-k through the single Triton router (`ops.moe` `moe_fused_gate`) via `srt/layers/moe/topk.py`, not through new per-model kernels.**
- Retired, and must not come back: AOT `moe_fused_gate`, `kimi_k2_moe_fused_gate`, JIT `grouped_topk` and `moe_fused_gate_jit`.
- A path that changes routing numerics ships behind an opt-in `SGLANG_OPT_*` flag declared in `environ.py`, with kernel parity tests and end-to-end accuracy. Production defaults stay unchanged.
- Detect: new `*fused_gate*`/`*grouped_topk*` sources, re-registration in `common_extension.cc`, model-name branches in `topk.py`, or default flips with no eval.
- Src: [#29771](https://github.com/sgl-project/sglang/pull/29771), [#29997](https://github.com/sgl-project/sglang/pull/29997), [#44690](https://github.com/sgl-project/sglang/pull/44690).

## 4. Diffusion

**D1. Import diffusion kernels only from the `sglang.kernels.ops.diffusion` facade, never from a submodule.**
- The facade is a PEP 562 lazy package, with `_EXPORTS` plus `_SPECS`.
- Why: the backends (Triton, CuTe-DSL, FlyDSL) have disjoint heavy dependencies, and the facade lets the layout move.
- The `test_import_surface.py` guard from #35114 was removed in #37436, so reviewers must enforce this by hand.
- Detect: `from sglang.kernels.ops.diffusion.<domain>.` in `multimodal_gen/`.
- Src: [#35114](https://github.com/sgl-project/sglang/pull/35114).

**D2. Lay out `ops/diffusion` by operator domain, with the backend as a filename suffix.**
- Domains: `norm`, `modulate`, `rope`, `activation`, `quantization`, `attention`, `routing`, `layout`.
- Suffixes: `_triton`, `_jit`, `_cutedsl`, `_flydsl`, `_bitexact`.
- `sites/` holds mount policy and is not kernels. `ext/` holds extensions. `common/platform.select_impl` is the only platform-dispatch seam.
- Detect: backend-named directories, import-time `if current_platform.is_X()` rebinding, or model-prefixed names on generic helpers.
- Src: [#35114](https://github.com/sgl-project/sglang/pull/35114), [#34180](https://github.com/sgl-project/sglang/pull/34180).

**D3. Gate every non-bit-exact fusion through `sites/` with a tier declared in `_QUALITY_FUSION_HANDLERS`.**
- `lossless` keeps the math and every operand's precision. `high` changes precision, sparsity or reuse. Never declare `exact`.
- Bit-exact fusions use `BitExactFusionGate.run()`.
- Why: the request tiers are fixed (`exact` < `lossless` < `high`), and per-model verify blocks kept drifting.
- Detect: new tier strings, per-model `_VERIFIED`/`_DISABLED` globals, mounting outside `sites/`, or a `lossless` claim without `quality_tier_admission.py`.
- Src: [#33453](https://github.com/sgl-project/sglang/pull/33453), [#34180](https://github.com/sgl-project/sglang/pull/34180), [#44690](https://github.com/sgl-project/sglang/pull/44690).

**D4. Put shared fallbacks and helpers in model-neutral modules.**
- Examples: residual/modulation, LN+modulate, QKV getters and native RMSNorm.
- Detect: near-identical helpers in two or more model files.
- Src: [#34085](https://github.com/sgl-project/sglang/pull/34085), [#34180](https://github.com/sgl-project/sglang/pull/34180).

## 5. Tests and CI

**T1. Put each test where the taxonomy says.**

| Test type | Location |
| --- | --- |
| Kernel numerics | `test/registered/kernels/ops/<group>/test_*.py` |
| Kernel benchmarks | `test/registered/kernels/benchmark/<group>/bench_*.py` |
| Inventory and dispatch | `test/registered/unit/kernels/` |
| Runtime unit tests | `test/registered/unit/<subsystem>/`; no server launch |
| End-to-end tests | sorted by kind: `e2e/models`, `accuracy/`, `perf/`, `stress/` |
| Diffusion workflow suites | bridge files in `e2e/diffusion/` |
| Non-CI scripts | `test/manual/` |

Further notes:
- Why: `check_registered_tests.py` enforces the kernel layout and the no-server rule for unit tests. `check_no_registered_tests_in_package.py` rejects tests inside `python/sglang`, which would ship but never run.
- Detect: files under the retired `test/srt/`, `test/registered/kernel/` or `test/registered/jit/`.
- Src: [#32128](https://github.com/sgl-project/sglang/pull/32128), [#39966](https://github.com/sgl-project/sglang/pull/39966), [#37436](https://github.com/sgl-project/sglang/pull/37436), [#41243](https://github.com/sgl-project/sglang/pull/41243).

**T2. Register every file at module level and make it run its own tests.**
- The call is `register_*_ci(est_time=<literal>, stage=..., runner_config=...)`. The legacy CUDA `suite=` form is rejected, except for stress.
- `__main__` must call `unittest.main()` or `pytest.main([__file__])`. Otherwise its TestCase classes are dead.
- Kernel tests use `*-kernel-*` suites.
- Hardware is expressed with extra `register_*_ci` calls, never with a directory.
- Src: [#39966](https://github.com/sgl-project/sglang/pull/39966), [#37436](https://github.com/sgl-project/sglang/pull/37436).

**T3. Register each test on a lane whose hardware satisfies its gates and whose image has its dependencies.**
- Why: #44690 found tests that never ran:
  - sm_100-gated tests on H100
  - multi-GPU diffusion tests on a 1-GPU lane
  - standalone files the planner never read
  - packages missing from the CI venv
- Detect:
  - An arch `skipif` that contradicts `runner_config`.
  - `disabled=` without an issue link and `until YYYY-MM-DD`.
  - A second accelerator registration without a `backend-specific:` comment.
- Src: [#44690](https://github.com/sgl-project/sglang/pull/44690), [#32128](https://github.com/sgl-project/sglang/pull/32128).

**T4. When you merge, split or move tests, keep every assertion, parametrization, skip and registration, and divide `est_time` between the parts.**
- Why: a merged file inherits the **union** of its sources' CI lanes. In #35114 that put CuTe cases on the AMD lane and FlashAttention cases on B200. Parts with disjoint platform needs stay in separate files with their original registration (`test_norm_flydsl.py`, `test_rope_ltx2.py`).
- Detect: a merged file that gains lanes some of its cases can't run on, or a total `est_time` that rises after a split.
- Src: [#35114](https://github.com/sgl-project/sglang/pull/35114), [#41243](https://github.com/sgl-project/sglang/pull/41243), [#40924](https://github.com/sgl-project/sglang/pull/40924).

**T5. Admit a test only if you can name the production change that would turn it red.**
- Reject:
  - tautologies, and echoes of defaults or log formats
  - source-shape tests: `inspect.getsource`, AST or regex mirrors, migration scanners
  - tests that only check call choreography through mocks
  - always-skipped tests
  - copied fixtures or reference code
- Mocks alone are not a reason to reject a test.
- Src: [#37436](https://github.com/sgl-project/sglang/pull/37436), [#39295](https://github.com/sgl-project/sglang/pull/39295), [#39293](https://github.com/sgl-project/sglang/pull/39293), [#40265](https://github.com/sgl-project/sglang/pull/40265).

**T6. Keep the PR lane cheap.**
- Use representative or pairwise cases in PR CI. The full sweep goes through `get_ci_test_range`, which only expands in nightly runs (`SGLANG_JIT_KERNEL_RUN_FULL_TESTS`).
- Each file should take under 500 s.
- Default registrations are capped at 1,200 weighted accelerator-seconds per backend, unless a `ci-cost-override:` comment says why.
- Write kernel tests as plain pytest functions with `parametrize` and `monkeypatch`.
- Src: [#31400](https://github.com/sgl-project/sglang/pull/31400) (535 → 32 cases), [#36887](https://github.com/sgl-project/sglang/pull/36887), [#31042](https://github.com/sgl-project/sglang/pull/31042), [#31546](https://github.com/sgl-project/sglang/pull/31546).

## 6. General cleanup

**G1. Declare env vars in `srt/environ.py` or `multimodal_gen/envs.py` with the `SGLANG_` prefix and a typed `Env*`, and read them through `envs`.**
- Why: code read legacy `SGL_*` names while the docs named `SGLANG_*`, so the documented knobs did nothing. `"1"` never matched `== "true"`, and twelve declared vars were never read.
- Detect: raw `os.getenv`/`os.environ` for these names, a declared var that nothing reads, or unprefixed dev knobs.
- Src: [#31042](https://github.com/sgl-project/sglang/pull/31042), [#44690](https://github.com/sgl-project/sglang/pull/44690).

**G2. Delete dead code completely, and give zero-consumer internals no shim.**
- "Completely" means the code, its flag, docs rows, tests, CI filters and registry entries.
- First check callers, including string targets and dynamic lookups.
- Only user-facing CLI or config surfaces get a deprecated alias.
- Detect: re-adding retired subsystems (`sglang.auto_benchmark`, PipeFusion plumbing, the no-op functionalization pass), a compatibility module for an internal API, or code marked "not wired yet".
- Src: [#31941](https://github.com/sgl-project/sglang/pull/31941), [#32651](https://github.com/sgl-project/sglang/pull/32651), [#39293](https://github.com/sgl-project/sglang/pull/39293), [#39295](https://github.com/sgl-project/sglang/pull/39295).

**G3. No commented-out code, `except Exception: raise`, `finally: pass`, shadowed duplicate definitions, unused imports or unused locals.**
- Detect: ruff `F401,F841,F811,F821,F541`.
- Src: [#29770](https://github.com/sgl-project/sglang/pull/29770), [#29789](https://github.com/sgl-project/sglang/pull/29789), [#30159](https://github.com/sgl-project/sglang/pull/30159).

**G4. A refactor PR proves it is equivalent and states its own limits.**
- It shows:
  - byte- or AST-identical moves
  - `torch.equal` or output hashes before and after
  - the same failing test IDs as on main
  - a list of platforms that were not validated
- Relocation and semantic commits are kept apart.
- A bug found along the way gets a regression test that fails on main.
- Detect: a "pure refactor" claim with kernel-body diffs, or a cleanup that loosens a threshold.
- Src: [#44690](https://github.com/sgl-project/sglang/pull/44690), [#45112](https://github.com/sgl-project/sglang/pull/45112), [#42391](https://github.com/sgl-project/sglang/pull/42391).

## Guards that fail automatically

| Guard | Catches |
| --- | --- |
| `test/registered/unit/kernels/test_kernels_namespace.py` | README vs `_GROUPS` and directories; stray `ops/*.py`; test groups; metadata-only import; stale targets; unregistered entry points; backend-directory pairing |
| `test/registered/unit/kernels/test_fused_op_dispatch.py` | `BaseFusedOp` priority ladder, platform chains, ops still on the deprecated alias |
| `scripts/lint/check_registered_tests.py` | missing or legacy registration, dead TestCase, kernel test layout, server launch in unit tests |
| `check_no_registered_tests_in_package.py`, `check_no_bare_pytest_main.py` | registered tests inside `python/sglang`; a bare `pytest.main` |
