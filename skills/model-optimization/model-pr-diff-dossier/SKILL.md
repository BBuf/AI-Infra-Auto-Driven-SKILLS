---
name: model-pr-diff-dossier
description: Use when creating or revising model PR optimization history documents for SGLang, vLLM, or another serving framework that cite GitHub PRs. Requires manual, per-PR source-diff review and documentation of motivation, key implementation approach, most important code excerpts, reviewed files, and validation implications instead of generated or one-line summaries.
---

# Model PR Diff Dossier

Use this skill before publishing any model PR optimization history document that
cites framework PRs.

## Non-Negotiable Standard

Do not summarize a PR with only a title-level sentence.
Do not use a script or bulk generator to fill motivation, implementation notes, or code excerpts.

For every PR cited as model optimization evidence, the document must include or link to a diff-reviewed PR card with:

- Immutable source head: the full repository commit used to trace model files
  and cross-check the PR. Record the capture date separately; never use `main`,
  a branch name, or a floating tag as the reproducibility anchor.
- PR link, title, state, merge time when available, additions/deletions, and changed-file count.
- PR-state classification: distinguish `open`, `merged`, and `closed-unmerged`.
  Open PRs are candidate evidence, not shipped behavior; record the reviewed
  head SHA and re-check state before publishing.
- Motivation: why the PR existed, inferred from PR body, title, issue context, docs changes, tests, and code diff.
- Key implementation idea: what runtime path changed and how the patch implements the change.
- Key code excerpts: short, relevant snippets from the actual diff, not invented pseudocode.
- Reviewed files: important files from the full diff, grouped by runtime/docs/tests where possible.
- Validation implications: tests, benchmark paths, launch flags, or regression lanes implied by the diff.
- Diff coverage note: state that the full diff was fetched/read and include diff line count.
- Limitations: gaps in public evidence, unverified hardware/model variants,
  open-PR instability, benchmark omissions, or other reasons the card should
  not be treated as a universal performance claim.

## Workflow

1. Resolve and record the target repository's immutable source head before
   tracing model files. Keep that head fixed for one dossier refresh.
2. Collect exact PR links from the target model history files. Use GitHub PR URLs, not bare `#123` text.
3. Query each PR's current state and head SHA, then classify it as `open`,
   `merged`, or `closed-unmerged`. Never describe an open PR as available in a
   released or recorded source head.
4. Open each PR diff directly with GitHub, `gh pr diff`, or the local framework repository commit. Read the changed source files, not just the PR title.
5. For merged PRs, cross-check the final code at the recorded source head when
   the diff is ambiguous. If the PR merged after that head, say so explicitly.
6. Write the PR card manually in the matching model history document. Use `references/card-schema.md` when you need the exact card shape. The card must name concrete files/functions/classes and include a short real code excerpt.
7. For docs-only or config-only PRs, quote the exact command/config line that changed and explain why it matters for serving or validation.
8. After each model family, review the cards for repeated shallow words such as "follow-up", "bugfix", or "optimization"; replace them with concrete implementation detail.
9. Record validation evidence and known limitations, then run repository tests
   and formatting before publishing.

## Review Gate

A model PR history is not ready if any PR card says only "follow-up", "bugfix", "docs", or "optimization" without:

- named files/functions/classes touched by the diff,
- a concrete motivation,
- a concrete implementation summary,
- at least one code excerpt or an explicit reason why the PR is docs-only,
- an immutable source head and correct open/merged/closed-unmerged state,
- validation evidence or a precise statement that no public validation exists,
- and explicit limitations on what the evidence proves.
