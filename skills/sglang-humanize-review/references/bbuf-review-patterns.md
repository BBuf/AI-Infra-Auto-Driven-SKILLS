# BBuf Review Patterns (sgl-project/sglang, 2026-07-27 to 2026-10-30)

Source: `bbuf-recent-review-corpus.jsonl.gz` (schema_version 2), a supplement to
`references/sglang-review-corpus.jsonl.gz` (which stops at 2026-07-27T15:49Z).
It holds 526 kept BBuf comments on 204 public PRs by other authors: 285 inline,
125 review-submission bodies and 116 PR-conversation comments. Review states on
those bodies: 63 APPROVED, 51 COMMENTED, 11 CHANGES_REQUESTED.

Counts below come from keyword heuristics run over the kept comments and were
spot-checked by hand. A comment can fall into more than one cluster, so treat
the numbers as approximate. Percentages are out of 526 comments.

## Recurring concerns

| # | Concern | ~Comments | ~PRs |
|---|---|---:|---:|
| 1 | State lifecycle, stale caches, memory retention | 73 (14%) | 46 |
| 2 | Tests that don't exercise the changed path | 66 (13%) | 56 |
| 3 | Simplify, reuse existing helpers, remove dead or duplicate code | 67 (13%) | 41 |
| 4 | Input contracts and fail-fast validation | 63 (12%) | 46 |
| 5 | Silent behavior change or compatibility break | 55 (10%) | 39 |
| 6 | Redundant or bloated tests; delete tests for trivial fixes | 53 (10%) | 41 |
| 7 | Scope: split unrelated or independent changes | 52 (10%) | 37 |
| 8 | CI coverage: registered, scheduled, actually executed | 51 (10%) | 41 |
| 9 | Accuracy or parity evidence on real data | 45 (9%) | 41 |
| 10 | PR description, naming, comments and docstrings | 38 (7%) | 32 |
| 11 | Benchmark validity | 37 (7%) | 30 |
| 12 | Housekeeping: lint, conflicts, rebase | 34 (6%) | 31 |
| 13 | Platform and arch gating (SM, HIP/ROCm, OOT) | 28 (5%) | 24 |
| 14 | Code placement (kernels/JIT tree, test directories) | 16 (3%) | 14 |
| 15 | No new flags or env vars; auto-enable instead | 10 (2%) | 9 |

### 1. State lifecycle and memory retention
He asks for per-request or per-sample resets and for caches keyed on the real
identity (effective config, file fingerprint). Memory must actually be released
on offload or weight update. When a path can't be made coherent, he wants online
weight updates rejected while it is enabled.
- "`prev` is never cleared between samples, so each prompt's step 0 is compared with the previous prompt's final step, corrupting the fit." ([#36767](https://github.com/sgl-project/sglang/pull/36767#discussion_r3893894860))
- "This reference keeps the packed CUDA storage alive after offload." ([#41339](https://github.com/sgl-project/sglang/pull/41339#discussion_r4111515749))

### 2. Tests that don't exercise the changed path
His most PR-diverse concern: does the new test reach the modified branch? Common
misses are mocks hiding the real kernel, eager-only tests for a graph-replay bug,
parameters that skip the new path, and assertions too weak to fail. He usually
names the case that would reach it.
- "This test mocks the actual BSA call, so it doesn't validate the new per-query block counts." ([#35850](https://github.com/sgl-project/sglang/pull/35850#discussion_r3843054087))
- "The current assertion only checks the NaN mask, so incorrect scales or zeroed finite outputs would still pass." ([#42689](https://github.com/sgl-project/sglang/pull/42689#discussion_r4213981663))

### 3. Simplify, reuse, remove dead code
He pushes back on abstractions with a single implementation, re-implemented
utilities, duplicated or unreachable blocks, helpers nothing calls, long
dispatch conditions, and hardcoded values.
- "Do we need the abstract base class here when HostExpertStore is the only implementation?" ([#42557](https://github.com/sgl-project/sglang/pull/42557#discussion_r4185116881))
- "The second copy is unreachable after the first one sets `text_encoder`; please remove it." ([#33569](https://github.com/sgl-project/sglang/pull/33569#discussion_r3746345301)); see also "Nothing on this branch calls any of it" ([#39664](https://github.com/sgl-project/sglang/pull/39664#issuecomment-5699199068)).

### 4. Input contracts and fail-fast validation
This shows up mostly on kernel wrappers. He wants dtype, device, contiguity,
alignment, stride and range checks, int32 overflow fixed, and nullptr
arithmetic avoided. Unsupported input should be rejected explicitly instead of
crashing or producing garbage. `can_use_*` predicates should return False and
never raise.
- "A bad caller can get silent garbage or an illegal access instead of a fail-fast error." ([#32327](https://github.com/sgl-project/sglang/pull/32327#discussion_r3746324925))
- "`can_use_*` should return `False`, not raise." ([#38182](https://github.com/sgl-project/sglang/pull/38182#discussion_r3947940175))

### 5. Silent behavior change or compatibility break
He flags paths that silently ignore input, fall back, override a documented
contract, or break existing callers (Python/CLI, other models, OOT platforms).
A warning is not enough for him: he wants the combination rejected or the old
behavior preserved.
- "a warning alone still returns approximate output under the lossless label." ([#36767](https://github.com/sgl-project/sglang/pull/36767#discussion_r3904549241))
- "We silently ignore every reference image after the first." ([#33386](https://github.com/sgl-project/sglang/pull/33386#discussion_r3802683813))

### 6. Redundant or bloated tests
Either a terse delete request (the fix is small and safe, or the test is a
one-off script), or a cleanup ask: consolidate into existing files and fixtures,
parameterize, cut long docstrings, aim for 2–3 files (CPU/kernel/e2e). These
comments grew from about mid-September, as large generated test files became
common.
- "The pr's change is small and safe, we don't need to add this test. Can you delete it?" ([#43640](https://github.com/sgl-project/sglang/pull/43640#discussion_r4259783756))
- "It's 253 lines for four cases, with quite a bit of repeated setup and explanation." ([#39889](https://github.com/sgl-project/sglang/pull/39889#issuecomment-5842692218)); "The goal is the smallest test set that catches the actual failures" ([#44887](https://github.com/sgl-project/sglang/pull/44887#issuecomment-6310538178)).

### 7. Scope: split unrelated changes
He asks for unrelated tweaks to be dropped, independent optimizations and opt-in
paths to go to follow-up PRs, overlap with already-merged PRs to be removed, and
the change made less invasive. He does not want one PR per bug, though: related
fixes stay together, each with its regression test.
- "Could we drop this unrelated tolerance change from this PR?" ([#32994](https://github.com/sgl-project/sglang/pull/32994#discussion_r3695669279))
- "Please split along those boundaries, keeping each regression with its fix; one PR per individual bug is unnecessary." ([#44309](https://github.com/sgl-project/sglang/pull/44309#pullrequestreview-5596043291))

### 8. CI coverage: registered and actually executed
`test/manual` files don't run in CI. Registered tests need the `__main__` entry
point, the right `runner_config` (for example an SM120 runner for SM120 code),
and the right `test/registered/...` directory. He also wants evidence that the
new GPU cases ran on the current head rather than being skipped. Many approvals
end with "please wait for the relevant (AMD/GPU) CI".
- "Could we move a focused fusion test to `test/registered`? `test/manual` is not covered by CI." ([#32994](https://github.com/sgl-project/sglang/pull/32994#discussion_r3695667716))
- "Therefore the file can run in CI, but the newly added SM120 admission is not actually exercised." ([#34327](https://github.com/sgl-project/sglang/pull/34327#issuecomment-5264760664))

### 9. Accuracy and parity evidence
For numerics, default-on flags or kernel swaps, he asks for real-dataset accuracy
(GSM8K, AIME, MMLU) with the flag on vs off. He also wants the results rerun on
the current head instead of an earlier revision, plus output or logprob parity.
- "Can you add some real datasets such as AIME or GMS8K's accuracy comparison results?" ([#33436](https://github.com/sgl-project/sglang/pull/33436#issuecomment-5199438786))
- "Since this is now default-on and changes the accumulation order, can we add a model-level accuracy comparison with the flag on and off?" ([#34331](https://github.com/sgl-project/sglang/pull/34331#discussion_r3764661208))

### 10. Description, naming, comments, docstrings
He wants the PR description to match the code (Risk section, scope claims), and
asks about unexplained renames. Names should carry scope (a `dllm_` prefix, or
the kernel's real purpose). Comments and docstrings should be short or removed,
and docs should point to the existing page.
- "The Risk section still says this is default-off, but the code and the rest of the description say default-on." ([#34331](https://github.com/sgl-project/sglang/pull/34331#pullrequestreview-4914340389))
- "Can you clean up the comments or delete the comments directly?" ([#40557](https://github.com/sgl-project/sglang/pull/40557#discussion_r4068171999))

### 11. Benchmark validity
Both arms must use the same graph mode, the baseline must be the real pre-PR
code, and shapes must match production. He also asks for other in-scope GPUs
(H100/H20), exact commands with SHAs, and numbers that exclude
already-merged optimizations.
- "Let's benchmark both with `use_cuda_graph=False`. The current comparison mixes graph replay and eager execution." ([#37903](https://github.com/sgl-project/sglang/pull/37903#discussion_r3963789174))
- "`_legacy` still runs the rewritten kernel, so it only restores the old launch parameters, not the old scale-loading behavior." ([#35760](https://github.com/sgl-project/sglang/pull/35760#discussion_r3826477180))

### 12. Housekeeping
These are one-liners: "Please fix lint", "Please fix conflict", "Could you rebase
this PR onto the latest main branch?", or a bare link to a failing CI job.

### 13. Platform and arch gating
He prefers exact capability checks to range arithmetic. He checks that HIP/MUSA
majors are not read as CUDA SM versions. ROCm/XPU dispatch must be preserved,
and OOT hooks should be authoritative with safe defaults.
- "`sm_version // 10 == 12` is broader than necessary and may unintentionally enable this path for future SM12x variants." ([#39482](https://github.com/sgl-project/sglang/pull/39482#issuecomment-5694697909))
- "On HIP this major is not a CUDA SM version, so large shapes can select the NVIDIA cp.async/TMA path." ([#34727](https://github.com/sgl-project/sglang/pull/34727#discussion_r3861513650))

### 14. Code placement
Kernels belong in the `sglang/kernels` tree, with JIT preferred over AOT
sgl-kernel. Model-specific logic stays in the model path, not shared
integration code. Runtime-integration tests go under
`test/registered/unit/layers/...`.
- "Can you remove this kernel to jit?" ([#38040](https://github.com/sgl-project/sglang/pull/38040#discussion_r4145875547)); "Can we move it to sglang/kernel?" ([#32219](https://github.com/sgl-project/sglang/pull/32219#discussion_r3662245831))

### 15. No new flags or env vars
If an optimization is safe, he wants it enabled automatically when applicable.
A default-on change needs accuracy evidence first.
- "Could we enable this optimization automatically when applicable, without adding another flag? Otherwise, `server_args` will keep growing." ([#32994](https://github.com/sgl-project/sglang/pull/32994#discussion_r3695671794))
- "Can we avoid add this environ?" ([#42929](https://github.com/sgl-project/sglang/pull/42929#discussion_r4361082342))

## Tone and format habits

- **Questions over commands.** About 43% of comments contain a "Could we / Can we
  / Should we …?" question, and 22% open with one. "Please …" shows up in 33%,
  mostly in summaries and follow-ups.
- **Brevity by default.** The median length is 266 chars inline, 224 for review
  bodies and 494 for conversation. 66 comments are 80 chars or less and 190 are
  200 or less. Test cleanup and housekeeping get one-liners such as "Can we delete
  this test?", "Please fix lint" and "Please clean up the test". The English in
  these is informal and unpolished ("Can we avoid add this environ?").
- **Evidence-first findings.** He states the concrete failure (shape, value,
  config, rank count) before the ask, e.g. "This should break TP=7: the layer
  expects 480 features and receives 3360." ([#33182](https://github.com/sgl-project/sglang/pull/33182#discussion_r3802683190)).
  He often reproduces it himself (25 comments): "I tried `topk=64` on H200: this
  check accepts it, then the `KU_ASSERT` calls `exit(1)` and kills the process."
  ([#32327](https://github.com/sgl-project/sglang/pull/32327#discussion_r3746324923)).
- **States what he did not validate.** 36 comments do this, e.g. "I haven't run
  the GPU tests." ([#42557](https://github.com/sgl-project/sglang/pull/42557#pullrequestreview-5416093875)),
  "CPU function checks only; no LingBot/GPU run." ([#44888](https://github.com/sgl-project/sglang/pull/44888#pullrequestreview-5596042787))
- **Severity tags.** `[P1]`/`[P2]` appear in 36 comments, all from mid-September
  onward. `Non-blocking:` appears in 27 (mostly October), and `nit:`/`nits:` is
  rare. Even blocking comments are usually phrased as requests rather than
  demands.
- **Review summary bodies are pointers.** COMMENTED and CHANGES_REQUESTED bodies
  are often one line that frames the inline comments: "A few cache edge cases."
  ([#34274](https://github.com/sgl-project/sglang/pull/34274#pullrequestreview-4924860684)),
  "Two input-contract edge cases below." In September about 7 long reviews used
  a "Change summary + Mermaid flowchart" layout.
- **Approvals carry a reason plus a residual ask.** Of 63 approvals with a body,
  29 start with "LGTM", usually followed by one sentence on why the fix is
  correct. 33 mention CI ("Let's wait for CI", "Please run the AMD CI before
  merging"), and he often adds a follow-up to keep ("Please keep that regression
  test"). He rarely requests changes: 11 of 125 bodies.
- **Hands-on maintainer actions.** About 68 comments report his own work:
  "Targeted validation for `<sha>`" with `/rerun-test`, independent GPU
  validation reports, pushed rebases or fixes, and fix PRs on the
  contributor's fork.
- **Language.** All comments in this window are in English.

## Implications for `sglang-humanize-review`

1. Check that each new or changed test reaches the modified branch: no mock in
   place of the real code, graph replay where the bug lives, inputs that hit the
   new path, and assertions that can fail.
2. Flag redundant or bloated tests. Suggest deleting tests for small, safe
   fixes, and otherwise consolidating them into existing files and fixtures.
3. Check CI wiring: `test/registered` placement, a `__main__` entry, a runner
   for the target arch, and proof the test ran on the current head.
4. Prefer auto-enabling over new flags or env vars, and require accuracy
   evidence for anything default-on.
5. Ask for unrelated changes to be split out, but keep related fixes together.
6. Audit benchmarks: same graph mode in both arms, a real pre-PR baseline,
   production shapes, commands and SHAs.
7. Use his register: a short question that states the concrete failure, a
   `[P1]`/`[P2]`/`Non-blocking` tag when it matters, a one-line review body,
   and a note on what was not validated.
