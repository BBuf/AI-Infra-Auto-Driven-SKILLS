# Profiler PR review — 2026-11-08

These are open proposals, not installed capabilities. Each card below follows
the full PR diff, body, issue comments, reviews and inline review comments at
the recorded head. Diff file counts match the paginated files API. Mainline
comparison uses SGLang `c24285bde97c3e6424cd5d8cb35788889034d8ea`:
[v1 scheduler profiler](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/managers/scheduler_components/profiler_manager.py),
[v2 profiler](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/observability/profile_utils.py)
and [native CLI](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/profiler.py).
Those three modules were read in full. Old PR paths under `srt/utils/` must be
mapped to the current `srt/observability/` implementation before applying a patch.

The client in this skill uses bounded v1 capture, rejects filename separators
in labels and attempts to stop capture after a failed probe. Neither the
source review nor the client CPU regressions are a live GPU capture result.
Mainline defaults are `with_stack=True`, `record_shapes=False` in both v1/v2.

## PR #47329 — Require bounded per-stage profiling

- Link: https://github.com/sgl-project/sglang/pull/47329
- State: open, unmerged; head `22e47399295b956458d91659f906573e2c0e3c97`,
  base `974831643fd5a4944b506439d4404ae64fe69539`.
- Diff coverage: full diff fetched/read, 72 lines, 2 files, +35/-1.
- Motivation: native stage capture without a step target reaches scheduler
  counters that were never initialized. Reject the request before scheduling.
- Implementation: `ProfileReq.__post_init__` rejects absent, zero and negative
  `num_steps` whenever `profile_by_stage` is true. The predicate also applies
  to a STOP-shaped request carrying that flag; a normal STOP remains valid.
- Key excerpt:

```diff
+    def __post_init__(self):
+        if self.profile_by_stage and (self.num_steps is None or self.num_steps <= 0):
+            raise ValueError("profile_by_stage requires a positive num_steps")
```

- Reviewed files: `srt/managers/io_struct.py` and
  `test/registered/unit/api/v1/test_parity.py` (paths below `python/sglang/`
  for runtime modules throughout this document).
- Validation: tests exercise direct construction and Pydantic JSON validation,
  including STOP-shaped payloads. The author reports CPU validation only;
  the sole review is an automated notice, not approval. This proposal conflicts
  in API semantics with the two unbounded-capture proposals below.

## PR #47475 — Initialize stage counters without a step target

- Link: https://github.com/sgl-project/sglang/pull/47475
- State: open, unmerged; head `25644e1b4098d89f7ba224bb2d1dbe29289ba2b0`,
  base `5d4a6f499e028405ecd4c8362c1e8b2443df1758`.
- Diff coverage: full diff fetched/read, 99 lines, 2 files, +32/-15.
- Motivation: preserve an optional step limit while preventing v1's `None += 1`
  and v2's comparison with `None`.
- Implementation: initialize v1 counters independently of `num_steps`, replace
  stale targets on reconfiguration and guard both stop comparisons. v2's
  `_StageBasedTrigger` accepts an optional target and still stops on a stage
  transition. Its public `manual_stop` remains unimplemented: the PR's inline
  reference to explicit stop does not establish a usable v2 stop endpoint.
- Key excerpt:

```diff
+            target_count = self.stage_configs[s.curr_stage].target_count
+            if (target_count is not None and s.curr_count > target_count) or (
+                stage != s.curr_stage
+            ):
```

- Reviewed files: `srt/managers/scheduler_components/profiler_manager.py` and
  the PR's `srt/utils/profile_utils.py`.
- Validation: the author describes an external stub harness, but no test file
  ships in this diff. Review comments concern CI activation/title checks and
  disabled automated review. Verify repeat sessions, phase transitions and
  explicit stop separately; do not infer v2 manual control from the v1 change.

## PR #47497 — Allow v1 stage capture without num_steps

- Link: https://github.com/sgl-project/sglang/pull/47497
- State: open, unmerged; head `951cf6385ca95a028ddb025388ada4b2889ffb7b`,
  base `5008726431693ef0883f83b230c08f77434e9b3b`.
- Diff coverage: full diff fetched/read, 134 lines, 2 files, +87/-8.
- Motivation: address the same scheduler-counter crash while retaining
  unbounded native stage capture.
- Implementation: initialize both counters for every stage session, clear
  missing stage targets, and guard auto-stop. Unlike #47475 this patch changes
  only v1. `if num_steps` also treats zero as an absent target, so its request
  contract differs from #47329's positive-step validation.
- Key excerpt:

```diff
+                if (
+                    self.profiler_target_prefill_ct is not None
+                    and self.profiler_prefill_ct > self.profiler_target_prefill_ct
+                ):
```

- Reviewed files: `srt/managers/scheduler_components/profiler_manager.py` and
  `test/registered/unit/managers/test_profile_by_stage_without_steps.py`.
- Validation: the added CPU test uses a stub prefill batch to check the first
  increment and start callback. It does not cover a decode transition, stale
  state from a prior session, explicit stop or GPU output. Discussion contains
  a request to run CI and automated notices, with no substantive approval.

## PR #47631 — Keep trace filenames inside output_dir

- Link: https://github.com/sgl-project/sglang/pull/47631
- State: open, unmerged; head `af9a8ce1cf4eb9707215131f2afff3acb3c601e6`,
  base `419d31a7b21073cb43617924168bc6c9483b0755`.
- Diff coverage: full diff fetched/read, 129 lines, 4 files, +81/-2.
- Motivation: `profile_prefix` and `profile_id` are joined directly into output
  names; a relative traversal or absolute prefix can escape the chosen root.
- Implementation: replace separators and NUL in both fields before the v1/v2
  split in `_init_profile`; sanitize the encoder profiler prefix separately.
  Empty/None values retain their fallback behavior. This is label sanitization,
  not a restriction on the explicitly requested output directory.
- Key excerpt:

```diff
+    for sep in {"/", "\\", os.sep, os.altsep, "\0"}:
+        if sep:
+            value = value.replace(sep, "_")
+    return value
```

- Reviewed files: `srt/disaggregation/encoder/server.py`,
  `srt/managers/scheduler_components/profiler_manager.py`, the PR's
  `srt/utils/profile_utils.py`, and
  `test/registered/unit/managers/test_profile_filename_sanitize.py`.
- Validation: CPU tests construct filenames and check relative/absolute
  prefixes, profile IDs, NUL and unchanged plain labels. They do not run a live
  profiler. Discussion asks for CI activation; no approval establishes a merge.
  This skill rejects such labels before network/filesystem work rather than
  depending on that pending server patch.

## PR #46626 — Ignore PREBUILT in v2 stage mapping

- Link: https://github.com/sgl-project/sglang/pull/46626
- State: open, unmerged; head `da1085ea7509f71db1367b466def78f5f1010768`,
  base `1596e88edaf5c62b69323cb3df8a7d1a460dc472`.
- Diff coverage: full diff fetched/read, 66 lines, 2 files, +48/-1.
- Motivation: `ProfileManager.step` maps the forward mode even without an
  active capture. v2 raises on a PREBUILT placeholder while v1 skips it.
- Implementation: map PREBUILT to `None`, returning before the stage trigger
  advances or ends the current capture.
- Key excerpt:

```diff
-    elif forward_mode.is_idle():
+    elif forward_mode.is_idle() or forward_mode.is_prebuilt():
         return None
```

- Reviewed files: the PR's `srt/utils/profile_utils.py` and
  `test/registered/unit/utils/test_profile_utils.py`.
- Validation: the added CPU test configures a real `ProfileManager` with no
  backend activities, starts decode, then checks PREBUILT leaves trigger state
  unchanged. The author explicitly did not find a current event loop passing
  PREBUILT to `run_batch`; normal decode admission converts it first. This is
  a missing mapping/defensive fix, not proof that every PD deployment crashes.
  The only discussion asks for review; no reviews were returned.

## PR #39501 — Python 3.14 argparse compatibility

- Link: https://github.com/sgl-project/sglang/pull/39501
- State: open, unmerged; head `af888cf8da2e1a2fbd924870678b2a55fac50ddc`,
  base `a25f213bc4ff557249bd32916e23211e9cedf0e9`.
- Diff coverage: full diff fetched/read, 48 lines, 1 file, +0/-6.
- Motivation: the native CLI supplies `type=bool` to `BooleanOptionalAction`,
  whose Python 3.14 constructor rejects that keyword before parsing arguments.
- Implementation: remove that unused argument from stage, CPU, GPU, MEM, RPD
  and merge flags. The generated `--no-*` behavior remains unchanged.
- Key excerpt:

```diff
         "--profile-by-stage",
         action=argparse.BooleanOptionalAction,
-        type=bool,
         default=False,
```

- Reviewed file: `python/sglang/profiler.py`; no tests or review discussion.
- Validation: parser construction and `--help` are sufficient to exercise this
  failure; GPU execution cannot validate a constructor that fails first. The
  inspected main still contains all six arguments. Use a compatible interpreter
  or the skill's HTTP helper, and verify the deployed CLI before recommending it.

## PR #39506 — Tear down after memory-snapshot failure

- Link: https://github.com/sgl-project/sglang/pull/39506
- State: open, unmerged; head `34091fe4676a3b09c97ffb5ec7fbd8d619156c43`,
  base `bdf8886ad386685b4d089c8e66725b9280533591`.
- Diff coverage: full diff fetched/read, 139 lines, 2 files, +106/-2.
- Motivation: `_dump_snapshot` can raise before history recording is disabled
  and scheduler profiler state is reset.
- Implementation: catch dump errors, always attempt to disable history,
  preserve the original failure if cleanup also raises, then return
  `ProfileReqOutput(success=False)` after normal teardown. The patch affects v1's
  MEM section; it does not make every earlier torch export/barrier operation or
  v2's `_ProfilerMemory.stop` exception-safe.
- Key excerpt:

```diff
+            finally:
+                try:
+                    torch.cuda.memory._record_memory_history(enabled=None)
+                except Exception as e:
+                    cleanup_failure = f"Failed to disable memory history: {e}"
```

- Reviewed files: `srt/managers/scheduler_components/profiler_manager.py` and
  `test/registered/unit/managers/scheduler_components/test_profiler_manager.py`.
- Validation: two mocked tests cover snapshot failure and simultaneous cleanup
  failure. The author did not run the registered test because a dependency was
  missing; compilation/stub checks are not that test. No discussion or reviews
  were returned. Rebase the test's old constructor/topology assumptions before
  using it with current main. This skill's CPU/GPU trace capture does not request
  MEM snapshots.
