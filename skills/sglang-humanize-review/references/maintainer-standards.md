# Maintainer Standards for SGLang Reviews (required reading)

These are the current review standards of SGLang maintainer BBuf, distilled
from three sources, all as of 2026-10-30:

- his refactor and cleanup PRs ([refactor-conventions.md](refactor-conventions.md)),
- the 526 review comments he left on 204 other authors' PRs after the main corpus
  cutoff ([bbuf-review-patterns.md](bbuf-review-patterns.md), raw episodes in
  `bbuf-recent-review-corpus.jsonl.gz`),
- the review requirements he gave his coding agents.

**Precedence.** When these standards and an older corpus precedent disagree,
follow these standards: they describe how the codebase and the review bar look
now. The corpus still decides subsystem-specific risk; these rules decide
structure, tests, evidence and tone.

## 1. Review order

1. Understand the PR: what it does, which code it changes, the risks, and which
   key validation is missing. On a re-review, map each earlier comment to the
   fix and its location, and judge whether the fix is reasonable.
2. For AI-generated PRs (most incoming PRs), check structure and tests before
   logic: does the code sit where the current layout says, and do the tests
   prove anything? Then correctness.
3. Check reachability first: if nothing calls the new path, it is dead code.
4. Give an explicit verdict on merge risk.

## 2. Scope and size

- Every touched file outside the feature (CI scripts, unrelated tests, unrelated
  tolerances) needs a stated reason; otherwise ask to revert it to main.
- Keep intrusion into the main runtime small and changes general. Question a
  backend, flag or optimization added for a single model.
- Push for the smallest diff and the fewest files. Ask to drop non-general
  optimizations that show no clear gain.
- Split unrelated or independent changes into follow-ups, but keep related
  fixes together, each with its regression test. Tiny near-duplicate PRs can be
  folded into one ([#44309](https://github.com/sgl-project/sglang/pull/44309#pullrequestreview-5596043291)).
- Ask for verbose, over-defensive code to be heavily simplified: single-
  implementation abstract base classes, re-implemented utilities, duplicated or
  unreachable blocks, helpers nothing calls, long dispatch conditions.

## 3. Placement and structure (details in refactor-conventions.md)

Check every new file and import against the current layout:

- Kernels live under `python/sglang/kernels`, are imported from
  `sglang.kernels.ops.<group>`, and are grouped by complete computation, never by
  model (no `ops/<model>/` directory). `sglang.jit_kernel`, `srt/**/triton_ops`
  and kernels defined under `srt/` or `multimodal_gen/` are retired. Prefer JIT
  over new AOT `sgl-kernel` code.
- Kernel integration code, including wrappers for external libraries, belongs in
  the kernels tree, not `srt/`. Kernels never import `sglang.srt`; runtime state
  is passed in as arguments.
- New public kernel entry points register a lazy `KernelSpec`; a new group
  updates `ops._GROUPS` and the kernels README in the same PR.
- Multi-backend operators subclass `BaseFusedOp` with a real `forward_native`;
  backend methods (`forward_<backend>`) and platform methods (`forward_cuda`,
  `forward_hip`, ...) stay separate.
- Input contracts (dtype, device, contiguity, shape, stride, alignment, int32
  range) are enforced in the kernel's host launcher and fail fast. `can_use_*`
  gates only answer "is this kernel applicable", return `False` rather than
  raise, and never restate operand contracts or swallow errors into a silent
  fallback.
- Diffusion code imports kernels only from the `ops.diffusion` facade; every
  non-bit-exact fusion goes through `sites/` with a declared quality tier.
- MoE gate/top-k goes through the single Triton router; retired per-model gate
  kernels must not come back.
- Moves use `git mv` plus import rewrites with no forwarding shim, and update
  every reference (patch strings, spec targets, tuning paths, CODEOWNERS, CI
  filters) in the same PR.

## 4. Flags, env vars and dispatch

- Do not add new environment variables, `SGLANG_DISABLE_*` kill switches or
  `server_args`. Enable a safe optimization automatically from GPU arch, shape
  and backend conditions, and limit dispatch to the measured winning range
  ([#32994](https://github.com/sgl-project/sglang/pull/32994#discussion_r3695671794)).
- If an env var is truly needed, it is declared in `srt/environ.py` or
  `multimodal_gen/envs.py` with the `SGLANG_` prefix and read through `envs`.
- A path that changes numerics stays opt-in until model-level accuracy evidence
  exists; default-on changes need accuracy with the flag on and off.
- Use exact capability checks, not range arithmetic (`sm_version // 10 == 12`
  is too broad); never read a HIP/MUSA major as a CUDA SM version; keep
  ROCm/XPU/OOT dispatch intact. An auto backend must not pick a configuration
  that startup later rejects.
- If invasive code is unavoidable, name it by its scope (for example `dllm_*`).

## 5. Tests

- Admit a test only if you can name the production change that would turn it
  red. Ask to delete: mock-only call-choreography tests, interface or
  attribute-existence tests, input-legality tests, tautologies and source-shape
  tests. Expect AI-generated test code to shrink by roughly 90%.
- Check that each new test reaches the modified branch: no mock in front of the
  real kernel or executor, CUDA-graph replay when the bug lives there,
  parameters that hit the new path, and assertions on values rather than masks
  or shapes ([#42689](https://github.com/sgl-project/sglang/pull/42689#discussion_r4213981663),
  [#35850](https://github.com/sgl-project/sglang/pull/35850#discussion_r3843054087)).
- For a small, safe fix, or a narrow kernel unlikely to change, ask to delete the
  new per-PR CI test ([#43640](https://github.com/sgl-project/sglang/pull/43640#discussion_r4259783756)).
  Do keep the regression test that reproduces a fixed bug, and keep meaningful
  existing assertions.
- Prefer extending existing test files and fixtures (parametrize, `subTest`)
  over new files; aim for a few files (CPU / kernel / e2e). Ask to shorten long
  generated test files ([#39889](https://github.com/sgl-project/sglang/pull/39889#issuecomment-5842692218)).
- Tests live in the registered layout (`test/registered/kernels/ops/<group>/`,
  `test/registered/unit/<subsystem>/`, `e2e/...`); `test/manual` does not run in
  CI. Registered files need a literal `register_*_ci(...)` call, a `__main__`
  that runs the tests, and a runner whose hardware matches the code's gates.
  Merged test files inherit the union of their sources' CI lanes.

## 6. Evidence in the PR description

- The description must match the current head (scope, default-on/off, Risk
  section) and include the exact launch and benchmark commands with base and
  head SHAs.
- Performance claims: measured on the hardware the PR targets, across several
  models and concurrency levels, against the strongest tuned baseline and SGLang
  main (not other frameworks), both arms in the same graph mode, production
  shapes, cold L2 on an uncontended GPU, and no credit for optimizations that
  already landed ([#37903](https://github.com/sgl-project/sglang/pull/37903#discussion_r3963789174),
  [#35760](https://github.com/sgl-project/sglang/pull/35760#discussion_r3826477180)).
- Keep only changes that improve both the kernel and end-to-end numbers; a
  kernel that regresses anywhere stays at main. Kernel PRs show microbenchmark
  and correctness results; a PR with no speedup should be closed.
- Numerics changes show real-dataset accuracy (GSM8K, AIME, MMLU) and, for
  replacement kernels, bitwise equality where feasible and unchanged speculative
  accept length. Rerun after a rebase; results from an earlier revision do not
  count.
- Diffusion PRs show kernel and end-to-end numbers plus rendered image or video
  comparisons against main.
- Refactor and cleanup PRs prove equivalence (byte/AST-identical moves, output
  hashes, the same failing tests as main) and list the platforms not validated.

## 7. CI coverage and the merge bar

- A PR is mergeable when the tests related to the change actually passed on the
  current head, every other failure is shown to be unrelated to the diff, there
  is no conflict, and it has an approval. Optional unrelated jobs do not need to
  be green.
- Skipped, cancelled, zero-test and `run-ci` gate-failed jobs are not passes.
  Ask for proof that new GPU cases executed on this head, linked to the job.
- Related tests are triggered together, only the files the change touches.
- When validation has started, the head is pinned; updating main is for an
  approved PR whose CI breaks on a stale base.

## 8. Writing review comments

- English, inline at the exact code location, short and plain: the concrete
  failure (shape, value, config, rank count), then the ask, usually as a
  question ("Could we ...?", "Can we delete this test?"). No AI flavor, no
  headings or filler.
- Severity tags: `[P1]` must fix, `[P2]` should fix, `Non-blocking:` for advice,
  `Nit:` rarely. Raise only what must change; the goal is approve plus CI.
- A review body is one line that points at the inline comments ("A few cache
  edge cases.").
- An approval says why it is correct in one sentence, adds a one-line TODO when
  something should follow, and names the CI still to wait for ("LGTM. ... Let's
  wait for the AMD CI.").
- Say what you did not validate ("CPU-only checks; no GPU run.").
- Housekeeping is a one-liner: "Please fix lint", "Please fix conflict", "Could
  you rebase this PR onto the latest main branch?".
