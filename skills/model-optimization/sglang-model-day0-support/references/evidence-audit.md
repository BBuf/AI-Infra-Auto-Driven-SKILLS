# Public Evidence Audit

## Contents

- [Evidence Record](#evidence-record)
- [Manual PR Card](#manual-pr-card)
- [Workflow](#workflow)
- [State Rules](#state-rules)
- [Validation Hierarchy](#validation-hierarchy)
- [Prohibited Automation](#prohibited-automation)

## Evidence Record

Use one machine-checkable line for every PR used as evidence:

```markdown
- Evidence: https://github.com/sgl-project/sglang/pull/12345 | state: merged | head: 0123456789abcdef0123456789abcdef01234567 | limitation: validated only on the hardware and workload described by the public PR
```

Allowed states are `merged`, `open`, and `closed-unmerged`. Always record the
reviewed 40-character head. For open evidence, state the capture date in the
surrounding prose and make the limitation explain that the behavior is not
merged or released.

## Manual PR Card

Every technical PR citation must include or link to a manually reviewed card
with:

- public link, title, state, creation/merge/close time;
- immutable reviewed head and capture date;
- additions, deletions, changed-file count, and complete diff coverage note;
- motivation derived from body, issue, discussion, code, tests, and docs;
- concrete implementation paths, symbols, state transitions, and fallbacks;
- a short real excerpt from the diff;
- important runtime, kernel, docs, and test files reviewed;
- validation evidence and the exact claim it supports;
- limitations: missing hardware, open state, absent end-to-end result, narrow
  shapes, or later changes.

Link to the existing model history card when it already satisfies this
contract. Do not duplicate long dossiers inside a case study.

## Workflow

1. Lock the public repository source head.
2. Query live PR state and reviewed head.
3. Fetch the complete diff and full file inventory.
4. Read motivation, discussion, code, tests, docs, and benchmark changes.
5. For merged PRs, cross-check final code at the locked mainline head.
6. Separate initial implementation from follow-up fixes, defaults, and reverts.
7. Write the card manually.
8. Add the machine-checkable evidence line.
9. Recheck live state immediately before publication.

## State Rules

- `merged`: describe only behavior reachable in the final reviewed mainline.
- `open`: describe candidate behavior at the immutable reviewed head; do not
  say available, landed, released, or supported without an explicit qualifier.
- `closed-unmerged`: use only as negative evidence or an abandoned approach.
- reverted work: classify `experiment-or-revert` even if the original PR
  merged.
- absorbed/superseded work: cite the surviving public implementation and note
  the supersession.

## Validation Hierarchy

Distinguish:

1. pure function or parser unit tests;
2. direct kernel parity;
3. dispatcher/envelope/fallback tests;
4. model-level deterministic generation;
5. protocol and serving tests;
6. accuracy evaluation;
7. memory/capacity validation;
8. end-to-end performance on a named hardware/workload;
9. nightly or release CI.

Do not upgrade a lower-level result into a higher-level claim.

## Prohibited Automation

Scripts may collect:

- metadata and state;
- immutable heads;
- file lists and diff statistics;
- public URLs;
- deterministic schema validation.

Scripts must not generate:

- motivation;
- implementation summaries;
- code excerpts;
- validation conclusions;
- performance explanations;
- Day-0 classification.

Those require manual diff review and final-mainline verification.
