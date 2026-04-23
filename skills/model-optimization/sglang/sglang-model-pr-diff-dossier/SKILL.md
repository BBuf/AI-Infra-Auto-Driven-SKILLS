---
name: sglang-model-pr-diff-dossier
description: Use when creating or revising SGLang model optimization skills or model PR optimization history documents that cite GitHub PRs. Requires manual, per-PR source-diff review and documentation of motivation, key implementation approach, most important code excerpts, reviewed files, and validation implications instead of generated or one-line summaries.
---

# SGLang Model PR Diff Dossier

Use this skill before publishing any SGLang model optimization skill or model PR optimization history document.

## Non-Negotiable Standard

Do not summarize a PR with only a title-level sentence.
Do not use a script or bulk generator to fill motivation, implementation notes, or code excerpts.

For every PR cited as model optimization evidence, the document must include or link to a diff-reviewed PR card with:

- PR link, title, state, merge time when available, additions/deletions, and changed-file count.
- Motivation: why the PR existed, inferred from PR body, title, issue context, docs changes, tests, and code diff.
- Key implementation idea: what runtime path changed and how the patch implements the change.
- Key code excerpts: short, relevant snippets from the actual diff, not invented pseudocode.
- Reviewed files: important files from the full diff, grouped by runtime/docs/tests where possible.
- Validation implications: tests, benchmark paths, launch flags, or regression lanes implied by the diff.
- Diff coverage note: state that the full diff was fetched/read and include diff line count.

## Workflow

1. Collect exact PR links from the target model skill and history files. Use GitHub PR URLs, not bare `#123` text.
2. Open each PR diff directly with GitHub, `gh pr diff`, or the local SGLang git merge commit. Read the changed source files, not just the PR title.
3. For merged PRs, cross-check the final mainline code in the local SGLang checkout when the diff is ambiguous.
4. Write the PR card manually in the matching model skill/history document. Use `references/card-schema.md` when you need the exact card shape. The card must name concrete files/functions/classes and include a short real code excerpt.
5. For docs-only or config-only PRs, quote the exact command/config line that changed and explain why it matters for serving or validation.
6. After each model family, review the cards for repeated shallow words such as "follow-up", "bugfix", or "optimization"; replace them with concrete implementation detail.
7. Run skill validation and repository formatting before publishing.

## Review Gate

A model PR history is not ready if any PR card says only "follow-up", "bugfix", "docs", or "optimization" without:

- named files/functions/classes touched by the diff,
- a concrete motivation,
- a concrete implementation summary,
- and at least one code excerpt or an explicit reason why the PR is docs-only.
