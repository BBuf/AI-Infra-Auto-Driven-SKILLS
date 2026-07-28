# Validation Matrix

## Risk Pairs

| Pair | Shared risk | Required check |
| --- | --- | --- |
| MTP + recurrent state | Accepted-token commit and rejected-token rollback | Accepted lengths zero, partial, and full versus target-only state |
| PD + recurrent state | Logical ownership and wire layout | Prefill export/decode import round trip across TP4 roles |
| CUDA Graph + padding | Sentinel slots and stale replay metadata | Mixed real/padded batch versus eager output and state |
| Streaming + tools | Marker fragmentation and end holdback | Split every marker at every byte boundary |
| VLM + packaging | Processor/runtime dependency skew | Cold-start image request in the release image |

## Required Lanes

| Gate | Lane | Procedure | Expected result |
| --- | --- | --- | --- |
| Source | Revision lock | Resolve every public revision to 40 hex characters | Exact match |
| Load | Eager BF16 | Load target and draft, audit keys, generate twice | No missing required keys; deterministic tokens |
| Protocol | Reasoning/tools | Stream all marker splits and compare non-streaming | Same public fields and tool arguments |
| State | MTP/PD/graph | Run the five risk-pair checks above | Reference-equivalent committed state |
| Topology | TP8 and TP4+TP4 PD | Include idle and uneven request batches | All ranks remain live and correct |
| Quality/performance | Fixed public corpus | Compare quality, memory, and eager latency | Quality in band and model fits |
| Release | Fresh image | Build, launch, run cookbook, scan bundle | Reproducible success |
