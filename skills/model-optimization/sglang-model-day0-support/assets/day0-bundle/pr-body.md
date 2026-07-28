# Public Pull Request Body

## Summary

Add Day-0 support for `{{MODEL_ID}}` at model revision
`{{MODEL_REVISION}}` and SGLang revision `{{SGLANG_SHA}}`.

Release cut:

- checkpoint/quantization: `{{CHECKPOINT_VARIANT}}` / `{{QUANTIZATION}}`;
- platform/topology: `{{PRIMARY_PLATFORM}}` / `{{TOPOLOGY}}`;
- protocol: `{{API_FEATURES}}`;
- speculative/multimodal: `{{SPEC_MODE}}` / `{{MULTIMODAL_MODE}}`.

## Implementation

- Configuration and loading: `{{LOADING_SUMMARY}}`
- Model, attention, MoE, and state: `{{MODEL_SUMMARY}}`
- Protocol and multimodal: `{{PROTOCOL_VLM_SUMMARY}}`
- Parallelism and platform: `{{TOPOLOGY_PLATFORM_SUMMARY}}`
- Fallbacks: `{{FALLBACK_SUMMARY}}`

## Validation

| Gate | Result | Public/reproducible evidence |
| --- | --- | --- |
| Source | `{{SOURCE_GATE}}` | `{{SOURCE_EVIDENCE}}` |
| Load | `{{LOAD_GATE}}` | `{{LOAD_EVIDENCE}}` |
| Protocol | `{{PROTOCOL_GATE}}` | `{{PROTOCOL_EVIDENCE}}` |
| State | `{{STATE_GATE}}` | `{{STATE_EVIDENCE}}` |
| Topology | `{{TOPOLOGY_GATE}}` | `{{TOPOLOGY_EVIDENCE}}` |
| Quality/performance | `{{QUALITY_GATE}}` | `{{QUALITY_EVIDENCE}}` |
| Release | `{{RELEASE_GATE}}` | `{{RELEASE_EVIDENCE}}` |

## Limitations

- `{{LIMITATION_ONE}}`
- `{{LIMITATION_TWO}}`

## Evidence

- Evidence: https://github.com/sgl-project/sglang/pull/{{PUBLIC_PR_NUMBER}} | state: {{PR_STATE}} | head: {{PUBLIC_HEAD_SHA}} | limitation: {{PUBLIC_EVIDENCE_LIMITATION}}

## Follow-up

Post-Day-0 fixes, performance work, platform extensions, experiments, and
reverts are tracked in `follow-up-ledger.md`.
