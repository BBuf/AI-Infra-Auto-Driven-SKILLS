# `<kernel-name>` Notes

> Copy this file into `AKO4ALL/<your-task>/context/<kernel>_notes.md`. Keep
> the contract small and explicit — every later iteration assumes these
> invariants.

## Contract

- Kernel entry point: `<file:line>` in target repo (and `<file:line>` in AKO copy)
- Inputs:
  - `<name>`: shape `<...>`, dtype `<bf16/fp16/fp32/fp8/int4/...>`, layout `<row-major/col-major/blocked/...>`, alignment `<bytes>`
  - ...
- Outputs:
  - `<name>`: shape, dtype, layout, alignment
- Strides: explicit / contiguous (state assumption)
- Mask / padding semantics:
- Numerical accumulation dtype: e.g. `fp32`
- Workload class: latency-sensitive / throughput-sensitive / both
- Target GPU architecture(s): sm89 / sm90 / sm100 / sm103 / sm120
- Shared-memory budget assumed: ___ KB / SM
- Register budget assumed: ___ / thread

## Reference Implementation

- File: `input/reference.py` (or `.cu`)
- How it was derived: PyTorch eager / previous operator / paper algorithm X / ...
- Tolerances accepted vs reference: max-abs ___ / max-rel ___ / dtype-aware

## Baseline Implementation

- File: `input/<kernel>.py` (or `.cu`)
- Source: copy of current implementation in `<target-repo>/<path>` at commit ___
- Known bottleneck (after baseline ncu/nsys): compute / memory / launch overhead / sync / layout-conv
- Profiler evidence:
  - ncu summary: <2-3 line summary>
  - nsys summary: <2-3 line summary>

## Production Shapes

| Tag | Shape | dtype | Layout | Notes |
| --- | --- | --- | --- | --- |
| hot | | | | first to optimize |
| tail | | | | small-shape edge case |
| ... | | | | |

## Failed Attempts (so they are not retried)

- Iter X: <hypothesis> → result <Δ%> → reason for rejection
- ...

## Open Questions

- ___
- ___
