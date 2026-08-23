# Kimi K3 Public Day-0 Case Study

## Contents

- [Public Evidence Boundary](#public-evidence-boundary)
- [Architecture Delta](#architecture-delta)
- [Day-0 Support Spine](#day-0-support-spine)
- [Immediate Public Follow-ups](#immediate-public-follow-ups)
- [Hardware and Packaging Extensions](#hardware-and-packaging-extensions)
- [Failure and Revert Lessons](#failure-and-revert-lessons)
- [Reusable PR Design Lessons](#reusable-pr-design-lessons)
- [Detailed Public Dossier](#detailed-public-dossier)

## Public Evidence Boundary

This case study uses only public `sgl-project/sglang` pull requests, their
public diffs, and public source paths. The audit snapshot is **2026-08-23**.
An open PR is evidence of proposed implementation, not evidence that support
has shipped. A merged documentation, image, or follow-up PR does not make an
open runtime spine release-ready by itself.

The principal runtime spine has now shipped:

- Evidence: https://github.com/sgl-project/sglang/pull/32541 | state: merged 2026-08-04 | released in SGLang `v0.5.17` (2026-08-08)
- Follow-ups that also shipped with that cut: DCP + DSpark `#32828`, standalone kernels `#32890`, reasoning/tool-call/OpenAI serving `#33025`
- Competitor landing: vLLM `v0.27.0` (2026-08-10) shipped a matching Kimi K3 stack (`#50089`, `#50000`, `#50093`, `#50090`, `#50458`, `#50500`, `#50242`)
- 2026-08-22 known issue in SGLang `v0.5.18`: the Kimi K3 MLA gate-projection fusion into the QKV-A GEMM landed and was reverted (`#33623`, `#34642`). Do not treat that fuse as current mainline behavior.

The reviewed `#32541` diff is unusually broad: it adds model and VLM
implementations, KDA and MLA paths, MoE routing, collectives, speculative
state handling, multimodal preprocessing, platform dispatch, tests, and
benchmarks. Keep using it as the support-spine lesson, not as a precedent
for accepting an unreviewable monolith on a later model.

## Architecture Delta

Kimi K3 is useful as a hard Day-0 example because its correctness surface is
larger than model registration and weight loading:

| Capability | Public implementation delta | Gate implication |
| --- | --- | --- |
| Hybrid sequence stack | Interleaved KDA and MLA layers require separate prefill, decode, metadata, and cache rules. | Exercise both layer families and boundary transitions. |
| Recurrent KDA state | Speculative verification must commit only accepted state; rejected draft tokens cannot leak. | Compare target-only and speculative state/output paths. |
| Latent MoE | Router scores, ranking bias, emitted weights, expert layout, and shared-expert overlap must agree. | Check router parity and non-fast-path fallback. |
| Quantization | MXFP4/W4A8 fast paths have shape, dtype, scale-layout, and hardware predicates. | Validate both selected and rejected dispatcher branches. |
| Distributed execution | TP, DP, EP, DCP, PP, PD/EPD, HiCache, and sequence-parallel paths change ownership and communication. | Test logical token/cache ownership, not only equal tensor sizes. |
| VLM | Image preprocessing, grid metadata, vision attention, and vision CUDA Graph reuse join the text runtime. | Add deterministic multimodal and invalid-media tests. |
| Protocol | Thinking, reasoning parsing, and tool-call auto-detection are observable API behavior. | Compare streaming and non-streaming responses. |

The key planning lesson is to turn each architectural novelty into a state
invariant and a fallback invariant before discussing peak throughput.

## Day-0 Support Spine

The open support spine adds `kimi_k3.py` and `kimi_k3_vl.py` plus the runtime
surfaces needed to execute them. Manual diff review identifies these
Day-0-required groups:

1. **Configuration and loading.** Register the model/config, map checkpoint
   names, load quantized experts, and construct the text and vision variants.
2. **KDA/MLA execution.** Provide KDA prefill and fused decode, MLA decode
   preparation, metadata construction, and safe fallback for uncovered shapes.
3. **MoE execution.** Preserve routing semantics while enabling fused MoE-front,
   radix top-k, SiTU expert kernels, and shared-expert overlap.
4. **State and memory.** Maintain recurrent KDA state, compressed or paged
   attention state, DCP locations, graph buffers, and accepted-token ReplaySSM
   semantics.
5. **Distributed paths.** Wire collective fusion, GEMM/all-gather,
   sequence-parallel collectives, and the advertised parallel modes.
6. **Multimodal execution.** Implement image normalization/patchification,
   vision metadata, backend selection, bounded graph caching, and eager
   fallback.
7. **API behavior.** Register reasoning and tool-call parsing, then test
   complete and incrementally streamed forms.
8. **Verification.** Supply kernel parity tests, dispatcher/fallback tests,
   model-level unit tests, VLM tests, and representative benchmarks.

The PR contains performance work that is valuable but not uniformly required
for first correct service. Classify launch fusion, specialized GEMMs,
communication fusion, and overlap as `performance-only` unless a documented
memory or latency constraint makes one necessary for the release cut.

## Immediate Public Follow-ups

These public PRs demonstrate why the support spine needs a follow-up ledger.
States are frozen at the audit date.

### Cookbook and release recipes

- Evidence: https://github.com/sgl-project/sglang/pull/32542 | state: merged | head: f795573d68a06b095c2605328c8fd000e05eda4b | limitation: documentation and launch recipes do not prove that the still-open runtime support spine has shipped

The cookbook adds hardware-specific launch and benchmark recipes. A later
documentation pass marks unsupported cells as in progress instead of
presenting every recipe as ready:

- Evidence: https://github.com/sgl-project/sglang/pull/32586 | state: merged | head: 8a21fdfbf445994d3ea696c4b1036222dde11b44 | limitation: recipe status is documentation metadata and must still be reconciled with runtime and image availability

### Public-source consistency

- Evidence: https://github.com/sgl-project/sglang/pull/32545 | state: merged | head: d6e00a35f212826358da3d8e5258c084ea926b8e | limitation: changing the Docker source fixes reproducibility only for the referenced public branch state
- Evidence: https://github.com/sgl-project/sglang/pull/32547 | state: merged | head: 47f77153fa841019ebdb8c11b3c6fc305e797abe | limitation: documentation links can become stale while the support branch and mainline continue to move

Together these small fixes expose a release rule: cookbook, Docker build
source, and runtime commit must resolve to the same public support state.

### Loading and protocol completion

- Evidence: https://github.com/sgl-project/sglang/pull/32563 | state: merged | head: a67f6d06ae24ebedf8854114b825e4cf87062e55 | limitation: remote object-store loading is covered for the changed loader paths, including the speculative draft path, not every storage backend
- Evidence: https://github.com/sgl-project/sglang/pull/32567 | state: open | head: e4fa8cea7487cb14012034896d9275cfd071e119 | limitation: proposed parser handling for an elided thinking close is not merged behavior
- Evidence: https://github.com/sgl-project/sglang/pull/32617 | state: merged | head: 4ae866ad25b6bf0be38929d977b23a614d7bd957 | limitation: auto-detection registration still requires end-to-end streaming and non-streaming API checks with the released tokenizer

These changes belong to the load and API gates. They should not be hidden in
a generic “model works” checkbox.

### KDA speculative and shape coverage

- Evidence: https://github.com/sgl-project/sglang/pull/32571 | state: merged | head: 488b2f247a459a75fd95327e9d95249311038e2d | limitation: KDA MTP verification optimization must preserve accepted-prefix state parity across draft lengths and fallback paths
- Evidence: https://github.com/sgl-project/sglang/pull/32624 | state: open | head: 24a2303b4e58362b74cd01ac043346b9bf8bd123 | limitation: wider TP head-count coverage is proposed and cannot be treated as released TP16 or TP32 support

The reusable risk pair is “recurrent state plus speculative acceptance.” Test
accepted lengths zero, partial, and full, then compare the committed state
against a target-only reference.

## Hardware and Packaging Extensions

Hardware coverage is a matrix, not a boolean:

- Evidence: https://github.com/sgl-project/sglang/pull/32568 | state: open | head: e9e7b2bd6540ed1d5caab751b5f00a58092b6cee | limitation: proposed AMD nightly accuracy jobs cover only their declared models, shapes, and runner environment
- Evidence: https://github.com/sgl-project/sglang/pull/32643 | state: merged | head: e369344a4d3233b3ab9869f90eb9aa0b90caa359 | limitation: publishing an AMD ROCm nightly image proves packaging, not model accuracy or performance
- Evidence: https://github.com/sgl-project/sglang/pull/32604 | state: open | head: 56207ca96dcfdbd107d45c7e9934cc3e24c04ba6 | limitation: the broad NPU port is open and requires separate review of model-specific changes versus shared runtime churn
- Evidence: https://github.com/sgl-project/sglang/pull/32630 | state: open | head: 7ce34e3c10420f27342a4491e70eb319a3a6403f | limitation: portable ROCm sampling fallbacks are proposed for specific DSpark and DFLASH paths and need distribution-level parity tests
- Evidence: https://github.com/sgl-project/sglang/pull/32650 | state: open | head: 2150cc999d3e9fd0500cd442b89f5624c653fa6b | limitation: parameterized SiTU FlashInfer MXFP4 selection remains open and must retain conservative dispatcher fallback
- Evidence: https://github.com/sgl-project/sglang/pull/32661 | state: open | head: 143ec36a60ff12c9b6d992641114eaed224c618b | limitation: draft VLM compatibility documentation does not establish runtime compatibility on its own

Keep platform image production, platform accuracy CI, kernel portability, and
documentation as separate PR nodes. This allows one platform to remain
experimental without blocking a correct and explicit primary release cut.

## Failure and Revert Lessons

Several earlier public VLM changes were closed without merging as standalone
PRs after their implementation was absorbed into the larger support spine.
That history should be recorded as **superseded**, not counted as multiple
independent fixes and not cited as shipped behavior. The absorbed themes were
vision projection copies, deterministic image seeds, CUDA IPC handle caching,
MoonViT grid-metadata placement, MoonViT RoPE/sequence bounds, and invalid
media validation.

This produces three useful review rules:

1. Re-check the final support-spine diff; do not assume every closed precursor
   survived unchanged.
2. Preserve focused regression tests even when implementation PRs are
   consolidated.
3. Prefer small prerequisite PRs for generic runtime fixes, leaving the model
   spine responsible for model wiring and model-specific contracts.

No public revert is used here to prove a K3 behavior. Open optimizations remain
`performance-only` or `experiment-or-revert` until their merge and validation
state changes.

## Reusable PR Design Lessons

- Build a **correctness spine** around configuration, loading, model forward,
  state, protocol, and one declared platform before adding the full fast-path
  ladder.
- Split broadly reusable runtime fixes from model-specific code. Merge generic
  prerequisites first and make dependency order visible.
- Require each specialized dispatcher to state shape, dtype, architecture, and
  fallback predicates. Test both sides of each predicate.
- Treat recurrent state, graph replay, distributed cache ownership, and
  multimodal preprocessing as state-machine problems.
- Keep cookbook commands synchronized with a public runtime source and a
  buildable image.
- Report optimization numbers only with the public workload, hardware, commit,
  and comparison scope that produced them.
- Make the release matrix honest: `merged`, `open`, `experimental`, and
  `unsupported` are different states.

## Detailed Public Dossier

For the full manually diff-reviewed history and file coverage, use:

- [English Kimi public PR dossier](../../../../model-pr-optimization-history/sglang/kimi/README.en.md)
- [中文 Kimi 公开 PR 档案](../../../../model-pr-optimization-history/sglang/kimi/README.zh.md)

Re-audit live PR state before reusing this case after 2026-08-23. The
follow-up items below the spine may still be open or already merged; query
GitHub at review time instead of copying the 2026-07-28 open/head snapshots.
