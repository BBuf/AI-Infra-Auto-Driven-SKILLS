# DeepSeek V4.1: Image-First Delivery and a Linear Support Stack

Public evidence snapshot: 2026-10-05. PR dates below use frozen-main squash
commit dates; historical GPU results remain attributed to their original PRs.
This refresh did not run GPU validation.

## Release Cut Before Mainline

[Cookbook #38802](https://github.com/sgl-project/sglang/pull/38802) landed
2026-09-09; NVIDIA/B300 cells were marked verified by #38839/#38861 on
2026-09-10. The [integration spine #38798](https://github.com/sgl-project/sglang/pull/38798)
asked users to use image commit `da64c5cb` while its branch was unstable.
Lock the image digest and source commit separately from an evolving PR head.
A verified cookbook cell proves only its declared hardware/model/image lane.

## Reviewable Extraction Stack

The monolithic branch was split into a linear stack, merged 2026-09-15–18:

| Capability | Extraction PR | Evidence rule |
|---|---|---|
| Standalone kernels and wrappers | [#39646](https://github.com/sgl-project/sglang/pull/39646) | Byte-identical files to snapshot `01d34d7b`, AST-identical Python functions, unchanged parity tests. |
| Top-k | [#39648](https://github.com/sgl-project/sglang/pull/39648) | Preserve ordering, ties and dtype contracts. |
| Compression, KV I/O and metadata | [#39652](https://github.com/sgl-project/sglang/pull/39652) | Verify page ownership and eager/graph metadata parity. |
| Communication | [#39653](https://github.com/sgl-project/sglang/pull/39653) | Preserve logical collective ownership. |
| RoPE and FP4 packing | [#39656](https://github.com/sgl-project/sglang/pull/39656) | Preserve numerical and physical layout contracts. |
| Hopper FP8 GEMMs | [#39657](https://github.com/sgl-project/sglang/pull/39657) | Record architecture predicates and fallback. |
| mHC | [#39664](https://github.com/sgl-project/sglang/pull/39664) | Preserve residual state at each boundary. |
| Chat and tool parsing | [#39665](https://github.com/sgl-project/sglang/pull/39665) | Validate incremental streaming and tokenizer revisions. |
| Engram | [#39666](https://github.com/sgl-project/sglang/pull/39666) | Lock table indexing and transfer contracts. |
| Vision tower | [#39668](https://github.com/sgl-project/sglang/pull/39668) | Validate processor, grid metadata and bounded graphs. |
| Candidate indexer | [#39671](https://github.com/sgl-project/sglang/pull/39671) | Validate sparse candidate selection. |
| Rust extensions | [#39677](https://github.com/sgl-project/sglang/pull/39677) | Prove extension ABI and packaging. |
| Remaining model/runtime integration | [#38798](https://github.com/sgl-project/sglang/pull/38798) | Merged 2026-09-18; original restack reproduced tree `d529d05f`. |

The first release containing this spine is v0.5.21. Mechanical extraction
proofs establish equivalence to a pinned snapshot, not model correctness on
new hardware. Keep original parity tests and require real dispatcher coverage.
A linear stack can be preferable to an artificial parallel DAG when every
extraction depends on the same evolving runtime cut.

## Post-Day-0 Ledger

Keep these separate from the support spine:

- #40637 (2026-09-22): chunked paged-MQA metadata in eager.
- #40805 (2026-09-24): Triton 3.8 / CUDA 13.4 compatibility.
- #39929 (2026-09-25): reasoning-effort budgets.
- #41345 (2026-09-27): HiCache layer-transfer waits.
- #42128 (2026-10-01): SWA page size with bounded replay.

AMD extension #41018–#41021 landed 2026-09-25–28, followed by
[#41308](https://github.com/sgl-project/sglang/pull/41308), gfx950 serving,
on 2026-09-30. Label this a platform lane rather than backdating NVIDIA Day 0.
Performance work #39704, #39957, #40431, #40556, #41251, #41657,
#41658, #41660 and #42273 belongs in a separate optimization ledger.

## Applying the Pattern

1. Lock source, image digest, model and processor before comparing lanes.
2. Split standalone kernels and wrappers from model integration.
3. Record byte/AST equivalence against an immutable snapshot in extraction cards.
4. Preserve fallback, state and protocol invariants through the restack.
5. Reconcile cookbook status with mainline and the first shipping image/tag.
6. Classify later fixes, platform ports and performance work separately.

Use [the PR dossier](../../../../model-pr-optimization-history/sglang/deepseek-v41/README.en.md)
for per-PR diffs. Current source: [text model](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/models/deepseek_v4.py),
[vision model](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/models/deepseek_v41_vit.py),
[cookbook](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx).
