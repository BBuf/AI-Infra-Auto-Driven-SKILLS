# Validation Matrix

## Risk Pairs

| Pair | Shared risk | Required check |
| --- | --- | --- |
| `{{FEATURE_A}}` + `{{FEATURE_B}}` | `{{STATE_OR_CONTROL_RISK}}` | `{{PAIR_TEST}}` |
| `{{FEATURE_C}}` + `{{FEATURE_D}}` | `{{OWNERSHIP_OR_PADDING_RISK}}` | `{{PAIR_TEST_2}}` |

Select pairs that share token count, state layout/lifetime, graph buffers,
collectives, role transfer, or parser markers.

## Required Lanes

| Gate | Lane | Command or procedure | Expected result | Evidence |
| --- | --- | --- | --- | --- |
| Source | Revision lock | `{{SOURCE_CHECK}}` | All revisions immutable | `{{SOURCE_RESULT}}` |
| Load | Eager short generation | `{{LOAD_COMMAND}}` | Deterministic output | `{{LOAD_RESULT}}` |
| Protocol | Streaming/tools | `{{PROTOCOL_COMMAND}}` | Reference-equivalent fields | `{{PROTOCOL_RESULT}}` |
| State | Cache/spec/graph | `{{STATE_COMMAND}}` | No rejected or padded state leaks | `{{STATE_RESULT}}` |
| Topology | `{{TOPOLOGY}}` | `{{TOPOLOGY_COMMAND}}` | Correct and live on every rank | `{{TOPOLOGY_RESULT}}` |
| Quality/performance | Accuracy and capacity | `{{QUALITY_COMMAND}}` | `{{QUALITY_THRESHOLD}}` | `{{QUALITY_RESULT}}` |
| Release | Build/docs/sanitize | `{{RELEASE_COMMAND}}` | Public bundle is reproducible | `{{RELEASE_RESULT}}` |
