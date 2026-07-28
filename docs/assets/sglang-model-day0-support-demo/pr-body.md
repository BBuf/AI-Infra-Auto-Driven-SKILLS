# Public Pull Request Body

## Summary

Add a review design for fictional `Aurora-Hybrid-70B-VL`: BF16 text/VLM
serving on B200 with TP8 unified and TP4+TP4 PD, reasoning/tools, and a
remote-loaded MTP draft.

## Implementation

- Register the config, text/VLM model, weight mapping, and eager hybrid path.
- Add a shared public loader contract for target and speculative draft.
- Transfer recurrent state by logical token ownership in PD.
- Hold fragmented reasoning/tool markers across stream chunks.
- Use eager fallback for unsupported graph padding and kernel shapes.
- Package vision processor dependencies in the public release image.

## Validation

| Gate | Result | Evidence |
| --- | --- | --- |
| Source | designed | Four immutable public source revisions are listed |
| Load | designed | Complete key audit and deterministic eager smoke |
| Protocol | designed | Every marker split plus streaming parity |
| State | designed | MTP acceptance, PD transfer, and graph-sentinel parity |
| Topology | designed | TP8 and TP4+TP4 PD including idle ranks |
| Quality/performance | designed | Fixed public corpus, memory fit, fallback latency |
| Release | designed | Fresh image, cookbook replay, and public sanitizer |

## Limitations

- This is a resolved demonstration bundle, not a claim that Aurora exists.
- AMD, NPU, quantized MoE, multi-image, and video are out of scope.

## Evidence

- Evidence: https://github.com/sgl-project/sglang/pull/32541 | state: open | head: f748ae35a26fbe1be98db09967ffb828658b821a | limitation: public open precedent for hybrid model decomposition only
- Evidence: https://github.com/sgl-project/sglang/pull/23882 | state: merged | head: 7978aa75e2c16db50f249aa25b9c5678abf6c7d2 | limitation: public merged precedent for a broad model and state support spine only
