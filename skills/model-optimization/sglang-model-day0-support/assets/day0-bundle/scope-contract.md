# Day-0 Scope Contract

## Release Cut

- Model: `{{MODEL_ID}}`
- Checkpoint revision: `{{MODEL_REVISION}}`
- SGLang revision: `{{SGLANG_SHA}}`
- Primary platform: `{{PRIMARY_PLATFORM}}`
- Target release date: `{{RELEASE_DATE}}`
- Owner: `{{PUBLIC_OWNER}}`

Day-0 means all required capabilities below close the seven gates for this
exact release cut. It does not mean every optimization or platform is ready.

## Required Capabilities

| Lane | Required value | Success criterion |
| --- | --- | --- |
| Checkpoint | `{{CHECKPOINT_VARIANT}}` | All expected weights load at the pinned revision. |
| Quantization | `{{QUANTIZATION}}` | Reference and serving outputs remain in the declared tolerance. |
| API | `{{API_FEATURES}}` | Streaming and non-streaming responses satisfy the public protocol. |
| Topology | `{{TOPOLOGY}}` | Accuracy, state, and liveness checks pass on every required role. |
| Speculative | `{{SPEC_MODE}}` | Target-only and accepted-state behavior agree. |
| Multimodal | `{{MULTIMODAL_MODE}}` | Processor and server produce reference-equivalent inputs and outputs. |

## Out of Scope

- `{{OUT_OF_SCOPE_PLATFORM}}`
- `{{OUT_OF_SCOPE_OPTIMIZATION}}`
- Any model, artifact, or command revision not listed in the release lock.

Unsupported combinations must fail fast or be documented as unsupported.
