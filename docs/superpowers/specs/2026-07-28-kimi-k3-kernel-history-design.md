# Kimi K3 Kernel Optimization History Design

## Purpose

Turn the reusable kernel-engineering lessons from SGLang's public Kimi K3
Day-0 work into durable repository knowledge. The result should help future
kernel and profiling work without exposing private development metadata or
presenting experiments as landed production behavior.

The only pull request cited for this update is the public SGLang Day-0 support
PR, [`sgl-project/sglang#32541`](https://github.com/sgl-project/sglang/pull/32541).
Private pull-request numbers, branch names, commit hashes, machines, users,
paths, and artifact identifiers are out of scope.

## Success Criteria

The work is complete when:

1. The Chinese and English SGLang Kimi histories contain matching, manually
   reviewed coverage of the public Day-0 PR.
2. The history explains reusable mechanisms, applicability, tradeoffs, and
   validation boundaries rather than merely listing kernel names.
3. The fusion and overlap catalogs expose the same lessons in framework-neutral
   form for profiler-driven optimization.
4. Reverted, disabled, closed-unmerged, or experimental paths are not described
   as landed wins.
5. No private repository identifier or machine-specific detail appears in the
   diff.
6. Repository documentation and dossier-quality tests pass.
7. The change is pushed as a focused draft pull request against `main`.

## Approaches Considered

### 1. Public umbrella card plus reusable catalog entries

Add one evidence-rich card for the public Day-0 PR to both Kimi histories, then
extract the generally useful fusion, communication, and overlap patterns into
the two profiler catalogs.

This is the chosen approach. It keeps a single public provenance anchor while
making the lessons discoverable both by model-family history readers and by
users diagnosing traces.

### 2. Standalone Kimi K3 kernel case study

Create a new long-form case-study document and link it from the histories.
This would provide more narrative room, but would duplicate the history and
catalog structures and create another document that can drift.

### 3. History-only update

Add the public PR card without touching the catalogs. This is the smallest
change, but it would leave the optimization knowledge model-specific and less
useful during framework-neutral profiling.

## Content Design

### Model history card

Insert the public Day-0 PR before the existing documentation-only Kimi K3
entries. The Chinese and English cards have equivalent facts and structure:

- public link, title, open state at the audit date, exact source head, changed
  line count, and file count;
- motivation and the serving bottlenecks addressed;
- concrete implementation grouped by launch/copy elimination, specialized
  compute kernels, collective/compute fusion, asynchronous overlap, linear
  attention and ReplaySSM, and vision preprocessing;
- representative public source paths and short excerpts where they explain a
  mechanism better than prose alone;
- reviewed-file inventory and validation evidence;
- full-diff coverage statement and an explicit warning that an open PR is not
  equivalent to a merged release.

The card will distinguish public end-to-end attribution from finer-grained
claims. Publicly reported batch-1 throughput and category-level gains may be
included with source links, but microbenchmark results will not be generalized
to other shapes or hardware.

### Fusion catalog

Add compact framework-neutral patterns for:

- removing launch and copy overhead around many small projections;
- fusing collective completion with residual and normalization work;
- combining column-parallel GEMM with multicast all-gather;
- fusing attention decode prologues, data movement, layout conversion, and
  quantization;
- single-pass radix top-k selection;
- fused vision padding, normalization, and patchification.

Each entry states the signal that makes it applicable and the primary
correctness guardrail. Model-specific implementation names are used only as
public evidence, not as universal prescriptions.

### Overlap catalog

Add patterns for:

- side-stream execution of independent KDA and MLA branches;
- overlapping shared-expert compute with routed-expert communication;
- using programmatic dependent launch only after producer writes are visible;
- reducing recurrent-state replay to one batched fold across layers;
- choosing overlap only when the hidden work fits available slack.

The catalog will also record CUDA Graph stream-topology, tensor-lifetime, and
rank-symmetric allocation constraints that can invalidate otherwise promising
overlap designs.

### Open-PR watch

Retain the existing public `#32541` entry and update it only if its state or
metadata is inconsistent with the audited snapshot. Do not create a second
entry.

## Evidence and Data Flow

```text
public PR metadata and exact public diff
                 |
                 +--> bilingual Kimi history card
                 |
public Day-0 engineering report
                 |
                 +--> attributed performance context
                 |
reviewed public implementation paths
                 |
                 +--> generic fusion and overlap catalog patterns
```

The public PR diff is the source of implementation truth. The public Day-0
report supplies higher-level performance attribution and engineering context.
Catalog entries are derived explanations and must remain conservative when the
public evidence does not establish a universal result.

## Sanitization and Integrity Rules

- Cite only `sgl-project/sglang#32541` as pull-request provenance.
- Do not mention private PRs, repositories, branches, commits, hosts, IP
  addresses, usernames, filesystem paths, or proprietary artifact names.
- Describe mechanisms and applicability rather than internal chronology.
- Label the public PR as open using an explicit audit date.
- Do not claim that reverted or disabled experiments landed.
- Keep conditional techniques conditional, including exact-shape vision CUDA
  Graph capture and hardware-specific routing.
- Distinguish direct-kernel tests, dispatcher tests, model-level validation,
  microbenchmarks, and end-to-end measurements.

## Validation

Run:

1. targeted repository tests for model-history dossier quality, open-PR watch,
   profiler documentation, links, and metadata;
2. Markdown and pre-commit checks available in the repository;
3. bilingual structure and public-link parity checks;
4. repository-wide searches for private PR identifiers and environment
   fingerprints;
5. a final staged-diff review for unsupported performance claims, duplicated
   watch entries, temporary files, and unrelated changes.

## Commit and Pull Request Strategy

Use three focused commits:

1. this approved design and the implementation plan;
2. the bilingual public PR history card;
3. the reusable fusion and overlap catalog updates plus validation-driven
   corrections.

Push `agent/add-kimi-k3-kernel-history` and open a draft pull request against
`main`. The pull-request body will summarize the public evidence, sanitization
boundary, affected knowledge surfaces, and exact validation commands.
