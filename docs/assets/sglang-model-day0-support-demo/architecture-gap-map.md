# Architecture Gap Map

## Capability Classification

| Capability | Audit finding | Class | Fallback |
| --- | --- | --- | --- |
| Protocol | streaming reasoning/tool marker fragmentation | `day0-required` | Hold incomplete markers until the next chunk or stream end |
| Draft loading | remote speculative draft loading | `day0-required` | Download both target and draft through the same public loader contract |
| PD state | recurrent-state transfer in PD | `day0-required` | Disable PD until logical state layout and ownership agree |
| Graphs | CUDA Graph padding sentinels | `day0-required` | Eager execution for padded or unsupported batches |
| VLM release | multimodal image packaging | `day0-required` | Public image must contain processor and vision dependencies |
| Operations | post-Day-0 ownership | `day0-required` | Named public ledger owner and gate-reopen rule |
| Fused recurrent kernel | Shape-specialized decode | `performance-only` | Reference recurrent implementation |

## Evidence

- Evidence: https://github.com/sgl-project/sglang/pull/32541 | state: open | head: f748ae35a26fbe1be98db09967ffb828658b821a | limitation: Kimi K3 is an open hybrid-model precedent and is not shipped evidence for Aurora
- Evidence: https://github.com/sgl-project/sglang/pull/23882 | state: merged | head: 7978aa75e2c16db50f249aa25b9c5678abf6c7d2 | limitation: DeepSeek V4 is a merged compressed-state precedent, not validation of Aurora
