# Sanitization Report

## Public Evidence

- Allowed repositories: `{{ALLOWED_PUBLIC_REPOSITORIES}}`
- Public PR metadata captured at: `{{CAPTURE_TIME}}`
- Every technical citation has a manual diff-review card: `{{CARD_AUDIT_RESULT}}`
- Open evidence includes immutable head and limitation: `{{OPEN_EVIDENCE_RESULT}}`
- Performance claims retain revision, hardware, and workload scope: `{{PERFORMANCE_SCOPE_RESULT}}`

## Denylist Result

- Uncommitted denylist source: `{{DENYLIST_SOURCE}}`
- Private repository/PR/branch/commit scan: `{{PRIVATE_VCS_SCAN}}`
- People, machine, network, and absolute-path scan: `{{IDENTIFIER_SCAN}}`
- Registry, artifact, trace, and experiment-ID scan: `{{ARTIFACT_SCAN}}`
- Secret-like token scan: `{{SECRET_SCAN}}`
- Final diff reviewed by: `{{REVIEWER}}`

No denylist contents may be committed to the public bundle.
