# Review Goal

Judge whether the work for `{{TASK_GOAL}}` is ready to stop.

Read:

- `plan.md`
- `progress.md`
- changed files in `{{REPO_ROOT}}`
- artifacts under `artifacts/`

## Required Checks

### Deliverables

- `{{DELIVERABLE_CHECK_1}}`
- `{{DELIVERABLE_CHECK_2}}`

### Validation Commands

Run or inspect the result of:

- `{{VALIDATION_COMMAND_1}}`
- `{{VALIDATION_COMMAND_2}}`

### Metric Thresholds

- `{{METRIC_RULE_1}}`
- `{{METRIC_RULE_2}}`

### Safety And Scope

- No out-of-scope files were changed without justification.
- No required artifact or log is missing.
- Any skipped case is explicitly allowed by `plan.md`.

## Partial-Finish Policy

Return `"ready": true` for a partial finish only if `plan.md` explicitly allows it and the evidence shows the allowed stop condition was reached.

Otherwise, missing coverage, missing artifacts, flaky metrics, or failed validations must keep `"ready": false`.

## Reviewer Output Contract

Return strict JSON only.
Do not return markdown.
Use this schema:

```json
{
  "ready": true,
  "summary": "one-sentence verdict",
  "missing": [],
  "next_action": "what the builder should do next"
}
```

Rules:

- `ready`: boolean
- `summary`: short verdict
- `missing`: array of blocking gaps; use `[]` when ready
- `next_action`: one concrete next move; use an empty string only when ready
