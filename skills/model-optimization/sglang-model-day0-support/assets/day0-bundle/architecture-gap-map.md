# Architecture Gap Map

## Capability Classification

| Capability | Existing SGLang path | Model delta | Class | Fallback | Owner |
| --- | --- | --- | --- | --- | --- |
| Configuration/loading | `{{CONFIG_PATH}}` | `{{LOADING_DELTA}}` | `day0-required` | `{{LOAD_FALLBACK}}` | `{{OWNER}}` |
| Attention/state | `{{ATTENTION_PATH}}` | `{{STATE_DELTA}}` | `day0-required` | `{{ATTENTION_FALLBACK}}` | `{{OWNER}}` |
| MoE/MLP | `{{MOE_PATH}}` | `{{MOE_DELTA}}` | `{{MOE_CLASS}}` | `{{MOE_FALLBACK}}` | `{{OWNER}}` |
| Protocol | `{{PARSER_PATH}}` | `{{PROTOCOL_DELTA}}` | `day0-required` | `{{PROTOCOL_FALLBACK}}` | `{{OWNER}}` |
| Speculative | `{{SPEC_PATH}}` | `{{SPEC_DELTA}}` | `{{SPEC_CLASS}}` | `{{SPEC_FALLBACK}}` | `{{OWNER}}` |
| Multimodal | `{{VLM_PATH}}` | `{{VLM_DELTA}}` | `{{VLM_CLASS}}` | `{{VLM_FALLBACK}}` | `{{OWNER}}` |
| Platform | `{{PLATFORM_PATH}}` | `{{PLATFORM_DELTA}}` | `{{PLATFORM_CLASS}}` | `{{PLATFORM_FALLBACK}}` | `{{OWNER}}` |

Use only `day0-required`, `post-day0-fix`, `performance-only`, or
`experiment-or-revert`.

## Evidence

- Evidence: https://github.com/sgl-project/sglang/pull/{{PUBLIC_PR_NUMBER}} | state: {{PR_STATE}} | head: {{PUBLIC_HEAD_SHA}} | limitation: {{PUBLIC_EVIDENCE_LIMITATION}}

For every row, link a manually reviewed PR card or record why no public
precedent applies.
