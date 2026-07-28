# Pull Request DAG

## Dependencies

| Node | Scope | Depends on | Evidence class | Merge state |
| --- | --- | --- | --- | --- |
| P0 | `{{PUBLIC_INFRASTRUCTURE_SCOPE}}` | none | `day0-required` | `{{P0_STATE}}` |
| P1 | `{{MODEL_SPINE_SCOPE}}` | P0 | `day0-required` | `{{P1_STATE}}` |
| P2 | `{{PROTOCOL_VLM_SCOPE}}` | P1 | `day0-required` | `{{P2_STATE}}` |
| P3 | `{{PLATFORM_PACKAGING_SCOPE}}` | P1 | `{{P3_CLASS}}` | `{{P3_STATE}}` |
| P4 | `{{VALIDATION_DOCS_SCOPE}}` | P1, P2, P3 | `day0-required` | `{{P4_STATE}}` |

Each node must have a public purpose, bounded diff, named owner, fallback, and
validation command. The umbrella PR may visualize integration but must not
hide unresolved dependencies.

## Merge Gates

| Node | Gate to merge |
| --- | --- |
| P0 | Unit tests for the reusable contract and existing-model regressions pass. |
| P1 | Eager load, reference correctness, state, and required topology lanes pass. |
| P2 | Streaming/non-streaming protocol and multimodal parity pass. |
| P3 | Image builds and the declared platform accuracy lane passes. |
| P4 | Release lock, public commands, evidence audit, and sanitization pass. |

Known follow-up: `{{KNOWN_FOLLOW_UP}}`.
