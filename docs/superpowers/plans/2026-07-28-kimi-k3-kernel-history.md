# Kimi K3 Kernel Optimization History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sanitized, public-evidence-backed account of the reusable Kimi
K3 kernel optimizations to the bilingual SGLang model history and profiler
fusion/overlap catalogs.

**Architecture:** Use the single public SGLang Day-0 pull request as the
provenance anchor, preserve matching Chinese and English history-card
structures, and extract only framework-neutral mechanisms into the two
profiler catalogs. Keep open-PR state explicit, performance attribution
source-bound, and private development metadata out of every committed file.

**Tech Stack:** Markdown, Git, GitHub CLI, Python `pytest`, pre-commit.

---

## File Map

- `model-pr-optimization-history/sglang/kimi/README.zh.md`: Chinese public PR
  evidence card, optimization mechanisms, limitations, and validation.
- `model-pr-optimization-history/sglang/kimi/README.en.md`: English card with
  the same evidence and structure as the Chinese version.
- `skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`:
  framework-neutral fusion opportunities and correctness guardrails.
- `skills/llm-torch-profiler-analysis/references/overlap-catalog.md`:
  framework-neutral stream, PDL, collective, and replay overlap patterns.
- `model-pr-optimization-history/open-pr-watch.md`: existing public Day-0 PR
  status entry; modify only if the audited public metadata is stale.

### Task 1: Freeze and record the public evidence snapshot

**Files:**

- Read:
  `model-pr-optimization-history/sglang/kimi/README.zh.md`
- Read:
  `model-pr-optimization-history/sglang/kimi/README.en.md`
- Read:
  `model-pr-optimization-history/open-pr-watch.md`
- Read:
  `skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
- Read:
  `skills/llm-torch-profiler-analysis/references/overlap-catalog.md`

- [ ] **Step 1: Query current public pull-request metadata**

Run:

```bash
gh pr view 32541 --repo sgl-project/sglang \
  --json number,title,state,isDraft,createdAt,updatedAt,baseRefOid,headRefOid,additions,deletions,changedFiles,url,mergeable,reviewDecision
```

Expected: public PR `#32541` is returned with an exact head SHA, state, changed
file count, and line statistics.

- [ ] **Step 2: Fetch the exact public head and derive the comparison base**

Run:

```bash
git -C /tmp/sglang-kimi-audit-20260728 \
  fetch public refs/pull/32541/head:refs/audit/public-32541-current
git -C /tmp/sglang-kimi-audit-20260728 \
  merge-base 8d6549bc4039d33635844495d86684677a4f0df8 \
  refs/audit/public-32541-current
```

Expected: the fetched public head matches the GitHub metadata, and the
merge-base command returns the comparison base used by the public PR.

- [ ] **Step 3: Review the complete file inventory and kernel-relevant diffs**

Run:

```bash
git -C /tmp/sglang-kimi-audit-20260728 diff --name-status \
  a23f6ea09032811f200a103a40ccd92d03fd5285 \
  refs/audit/public-32541-current
git -C /tmp/sglang-kimi-audit-20260728 diff --shortstat \
  a23f6ea09032811f200a103a40ccd92d03fd5285 \
  refs/audit/public-32541-current
```

Expected: the inventory covers all 320 public changed files and the shortstat
matches GitHub's additions and deletions.

- [ ] **Step 4: Confirm open-PR watch has one public entry**

Run:

```bash
rg -n "sgl-project/sglang/pull/32541" \
  model-pr-optimization-history/open-pr-watch.md
```

Expected: exactly one entry exists. Preserve it unless its public state or
title is inconsistent with the fresh query.

### Task 2: Add the bilingual public Day-0 history card

**Files:**

- Modify:
  `model-pr-optimization-history/sglang/kimi/README.zh.md`
- Modify:
  `model-pr-optimization-history/sglang/kimi/README.en.md`
- Test:
  `tests/test_model_pr_dossier_quality.py`

- [ ] **Step 1: Add the Chinese card before the documentation-only K3 cards**

Add an `OPEN` card for `[Kimi] Support kimi-k3` with:

- exact public URL, audit date, head SHA, `+57450/-1534`, 58,984 changed lines,
  and 320 files;
- motivation covering launch overhead, memory traffic, small-shape compute,
  communication critical path, recurrent-state preparation, and vision
  preprocessing;
- concrete sections for launch/copy elimination, Blackwell-specialized
  compute, collective/compute fusion, stream and PDL overlap, KDA
  prefill/decode/MTP and ReplaySSM, and VLM preprocessing;
- reusable applicability and guardrails, including symmetric allocation,
  producer visibility, CUDA Graph topology, DCP logical locations, numerical
  semantics, and exact-shape capture;
- representative reviewed public source paths, test paths, public performance
  attribution, and a full-diff coverage statement.

- [ ] **Step 2: Add the matching English card**

Mirror every factual field and section from the Chinese card. Translate the
explanation naturally while preserving the exact URL, SHA, statistics, public
source paths, validation categories, and open-state warning.

- [ ] **Step 3: Verify bilingual evidence parity**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("model-pr-optimization-history/sglang/kimi")
docs = {
    name: (root / name).read_text(encoding="utf-8")
    for name in ("README.zh.md", "README.en.md")
}
needles = (
    "https://github.com/sgl-project/sglang/pull/32541",
    "ac6d795427cb9f0d149a8c318cbfcd4efa3aa62a",
    "58,984",
    "320",
)
for needle in needles:
    assert all(needle in text for text in docs.values()), needle
assert all(text.count(needles[0]) == 1 for text in docs.values())
print("bilingual public-evidence parity: ok")
PY
```

Expected: `bilingual public-evidence parity: ok`.

- [ ] **Step 4: Run the dossier-quality tests**

Run:

```bash
python3 -m pytest -q tests/test_model_pr_dossier_quality.py
```

Expected: all dossier-quality tests pass.

- [ ] **Step 5: Commit the bilingual history**

Run:

```bash
git add \
  model-pr-optimization-history/sglang/kimi/README.zh.md \
  model-pr-optimization-history/sglang/kimi/README.en.md
git commit -m "docs: add public Kimi K3 kernel history"
```

Expected: one commit containing only the paired history files.

### Task 3: Extract reusable fusion and overlap patterns

**Files:**

- Modify:
  `skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
- Modify:
  `skills/llm-torch-profiler-analysis/references/overlap-catalog.md`
- Test:
  `tests/test_llm_torch_profiler_analysis.py`

- [ ] **Step 1: Extend the fusion catalog**

Add compact rows for:

- grouped skinny projections and launch/copy elimination;
- single-pass register radix top-k;
- collective finalize plus residual/RMSNorm;
- column-parallel GEMM plus multicast all-gather;
- decode-prologue layout/cast/quantization fusion;
- fused vision pad/normalize/patchify.

For every row, include an applicability signal, expected trace change, and a
correctness or portability guardrail.

- [ ] **Step 2: Extend the overlap catalog**

Add compact rows for:

- independent KDA/MLA branch side streams;
- shared-expert compute versus routed-expert communication;
- producer-visible programmatic dependent launch;
- one batched recurrent-state fold across layers;
- overlap-slack accounting.

Add caveats for fixed CUDA Graph stream topology, tensor lifetime, logical
segment symmetry across ranks, and unconditional graph replay.

- [ ] **Step 3: Run profiler documentation tests**

Run:

```bash
python3 -m pytest -q tests/test_llm_torch_profiler_analysis.py
```

Expected: all profiler-analysis tests pass.

- [ ] **Step 4: Commit the catalog extraction**

Run:

```bash
git add \
  skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md \
  skills/llm-torch-profiler-analysis/references/overlap-catalog.md
git commit -m "docs: catalog reusable Kimi K3 kernel patterns"
```

Expected: one commit containing only the two catalog files.

### Task 4: Validate sanitization, repository quality, and PR scope

**Files:**

- Verify:
  `docs/superpowers/specs/2026-07-28-kimi-k3-kernel-history-design.md`
- Verify:
  `docs/superpowers/plans/2026-07-28-kimi-k3-kernel-history.md`
- Verify all files modified in Tasks 2 and 3.

- [ ] **Step 1: Search for forbidden private identifiers**

Run repository-wide searches for the known private repository namespace,
private pull-request URL pattern, machine addresses, user-specific paths, and
private artifact vocabulary. Inspect every match and require zero matches in
the branch diff.

Expected: no private identifier appears in any committed addition.

- [ ] **Step 2: Validate public links and unique watch state**

Run:

```bash
python3 -m pytest -q \
  tests/test_open_pr_watch.py \
  tests/test_repository_metadata.py
rg -n "sgl-project/sglang/pull/32541" \
  model-pr-optimization-history/open-pr-watch.md
```

Expected: tests pass and the watch contains one public PR entry.

- [ ] **Step 3: Run targeted and repository formatting checks**

Run:

```bash
python3 -m pytest -q \
  tests/test_model_pr_dossier_quality.py \
  tests/test_llm_torch_profiler_analysis.py \
  tests/test_open_pr_watch.py \
  tests/test_repository_metadata.py
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
git diff --check origin/main...HEAD
```

Expected: all tests and hooks pass, and `git diff --check` produces no output.

- [ ] **Step 4: Review exact branch scope**

Run:

```bash
git status --short
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

Expected: only the design, plan, paired history files, and two catalog files
are present; no temporary artifact or unrelated edit is included.

### Task 5: Publish the draft pull request

**Files:**

- No additional repository files.

- [ ] **Step 1: Push the focused branch**

Run:

```bash
git push -u origin agent/add-kimi-k3-kernel-history
```

Expected: the branch is published to `BBuf/AI-Infra-Auto-Driven-SKILLS`.

- [ ] **Step 2: Create a draft pull request**

Create a draft PR targeting `main` with a title that describes the public Kimi
K3 kernel-history extraction. The body must cover:

- what changed and why;
- the single public upstream provenance anchor;
- sanitization decisions and excluded experimental work;
- validation commands and outcomes;
- the fact that upstream `#32541` was still open at the audit timestamp.

Expected: GitHub returns a draft PR URL in
`BBuf/AI-Infra-Auto-Driven-SKILLS`.

- [ ] **Step 3: Verify the published PR**

Run:

```bash
gh pr view --repo BBuf/AI-Infra-Auto-Driven-SKILLS \
  --json url,title,state,isDraft,baseRefName,headRefName,statusCheckRollup
```

Expected: the PR is open, draft, based on `main`, and uses
`agent/add-kimi-k3-kernel-history` as its head.
