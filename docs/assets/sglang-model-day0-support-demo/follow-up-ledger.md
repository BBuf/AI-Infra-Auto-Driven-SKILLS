# Post-Day-0 Follow-up Ledger

## Open Fixes

| Item | Affected claim | State | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| None at design freeze | None | clear | public model-support working group | Reopen a gate when a claimed lane regresses |

The public working group owns triage for fourteen days after release.

## Performance Work

| Item | Correct fallback | Scope | State |
| --- | --- | --- | --- |
| Fused recurrent decode | Eager recurrent path | B200, batch one decode | planned |
| Quantized MoE | BF16 experts | Not in Day-0 release cut | deferred |
| Vision CUDA Graphs | Eager vision encoder | Repeated exact image grid only | planned |

## Experiments and Reverts

| Item | Effective shipped state | Evidence | Revisit condition |
| --- | --- | --- | --- |
| Graph padding fast path | Disabled for padded batches | Eager parity lane | Sentinel proof and dispatcher tests pass |
| Multi-image requests | Unsupported | Fail-fast API test | Processor and scheduler contracts land |

Open, closed-unmerged, reverted, and default-disabled work never expands the
release claim.
