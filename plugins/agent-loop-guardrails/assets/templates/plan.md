# Objective

Implement or complete `{{TASK_GOAL}}`.

## Working Context

- Repo root: `{{REPO_ROOT}}`
- Working directory: `{{WORKDIR}}`
- Main files or modules in scope:
  - `{{PATH_1}}`
  - `{{PATH_2}}`
- Reference docs, commands, or examples:
  - `{{REFERENCE_1}}`
  - `{{REFERENCE_2}}`

## Required Deliverables

- Code, docs, or scripts to produce:
  - `{{DELIVERABLE_1}}`
  - `{{DELIVERABLE_2}}`
- Artifacts to save on disk:
  - `{{ARTIFACT_1}}`
  - `{{ARTIFACT_2}}`

## Non-Goals

- `{{NON_GOAL_1}}`
- `{{NON_GOAL_2}}`

## Constraints

- Preserve `{{COMPATIBILITY_CONSTRAINT}}`.
- Do not modify `{{OUT_OF_SCOPE_PATHS}}` unless the plan is updated.
- Allowed commands:
  - `{{COMMAND_1}}`
  - `{{COMMAND_2}}`
- Environment or hardware assumptions:
  - `{{ENVIRONMENT_1}}`
  - `{{ENVIRONMENT_2}}`

## Execution Plan

1. Build context from the listed files and references.
2. Implement the smallest change that can move the task forward.
3. Run the required validations and write logs or artifacts to disk.
4. Update `progress.md` with the outcome, blockers, and next action.
5. Repeat until the success condition or hard stop condition is reached.

## Required Validations

- `{{VALIDATION_COMMAND_1}}`
- `{{VALIDATION_COMMAND_2}}`
- Quantitative thresholds:
  - `{{METRIC_1}}`
  - `{{METRIC_2}}`

## Success Condition

The task is complete only when all of the following are true:

- `{{SUCCESS_RULE_1}}`
- `{{SUCCESS_RULE_2}}`

## Acceptable Partial Finish

If full success is not possible, the task may still stop as acceptable only when:

- `{{PARTIAL_FINISH_RULE_1}}`
- `{{PARTIAL_FINISH_RULE_2}}`

Delete this section if the task must be strictly all-or-nothing.

## Hard Stop Conditions

Stop and escalate to the human when:

- `{{HARD_STOP_1}}`
- `{{HARD_STOP_2}}`

## Builder Notes

- Keep logs under `artifacts/attempt-*`.
- Update `progress.md` after each meaningful attempt.
- If a validation fails, record the exact command, failure summary, and next move.
- Do not declare success without the required validations.
