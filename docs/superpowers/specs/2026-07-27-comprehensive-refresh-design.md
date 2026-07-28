# Comprehensive AI Infrastructure Skills Refresh Design

## Purpose

Refresh the repository against its current July 27, 2026 upstream state, repair
correctness gaps discovered while reading every skill and executable, and make
the repository's generated evidence, operational guidance, tests, and
automation agree with one another.

The refresh must preserve the repository's central contract: each skill should
provide concise operational guidance, while deterministic scripts and
source-linked references carry fragile or high-volume work. It must not turn
skills into general documentation dumps or claim support that cannot be
verified.

## Success Criteria

The work is complete when:

1. All twelve skills and every bundled executable remain internally consistent
   with current SGLang, vLLM, TensorRT-LLM, TokenSpeed, CUDA, Claude Code, and
   Codex interfaces relevant to their documented workflows.
2. Known correctness bugs in trace analysis, capacity analysis, compute
   simulation, incident replay, model-history generation, and corpus
   generation have regression tests.
3. Source-derived artifacts record current immutable upstream heads and can be
   reproduced using documented commands.
4. Model histories include every model family for which the current upstream
   repositories contain meaningful evidence; speculative or empty histories
   are not added.
5. Open-PR watch data, profiler catalogs, architecture sources, review evidence,
   serving cookbook data, manifests, dependencies, and CI are current as of the
   refresh date.
6. Local tests, artifact validators, stale-reference checks, representative
   generation commands, and the repository's prescribed remote GPU checks pass.
7. The final branch is pushed and a draft pull request contains the complete
   change, validation evidence, and any explicitly documented limitations.

## Chosen Approach

Use a comprehensive, evidence-first refresh rather than a data-only update or a
large structural rewrite.

- Keep existing public CLI and skill boundaries unless a current upstream
  interface makes them incorrect.
- Add focused helpers where they make parsing or validation independently
  testable, but avoid reorganizing the repository wholesale.
- Write failing regression tests before each behavioral fix.
- Regenerate source-derived artifacts only after their generator and source
  classifications are correct.
- Treat immutable commits, merged pull-request diffs, official documentation,
  actual CLI help, and executed validation as authoritative evidence.

This approach updates everything that is demonstrably stale while minimizing
unrelated churn and preserving compatibility for existing skill users.

## Architecture and Component Boundaries

### 1. Deterministic Analysis Tools

The analysis scripts remain the executable core of their owning skills:

- `llm-pipeline-analysis` identifies forward-pass structure, anchors,
  representative layers, steady-state windows, and comparisons. Thresholds
  must derive from trace evidence instead of a fixed eight-millisecond
  assumption. JSON output must remain machine-readable even when comparison
  options are enabled, and event classification must be deterministic.
- `llm-serving-capacity-planner` recognizes both documented SGLang and vLLM
  startup evidence. A framework-specific parser may leave a metric unknown when
  the log does not prove it; it must not silently treat a vLLM log as SGLang.
- `model-compute-simulation` incorporates measured kernel-flow timing before it
  calculates summary MFU. Trace names and semantic template operations are
  normalized through an explicit mapping before missing/extra-operation
  comparisons are reported.
- `llm-torch-profiler-analysis` keeps its parser, source map, framework
  detection, and optimization catalog synchronized with current upstream
  layout. SGLang references move from the retired `sglang.jit_kernel` package to
  `sglang.kernels`; closed-unmerged optimization attempts are not represented
  as landed or in flight.
- `sglang-prod-incident-triage` validates replay speed and parallelism, surfaces
  HTTP failures before decoding responses, and documents the current HiCache
  storage-clear endpoint alongside attach and detach operations.

Each behavior receives a narrow unit test at the same abstraction boundary as
the fix. User-facing text remains stable where the semantics have not changed.

### 2. Upstream Evidence and Generated Artifacts

Four clean upstream checkouts provide the source snapshot:

- SGLang
- vLLM
- TensorRT-LLM
- TokenSpeed

The refresh records exact full commit SHAs, not mutable branch names alone. The
data flow is:

```text
upstream commits and PR diffs
        |
        +--> model-history generator --> per-model histories and indexes
        +--> open-PR watcher ---------> open-pr-watch.json
        +--> profiler audit ----------> source map and optimization catalog
        +--> review collector --------> review corpus and generated summaries
        +--> architecture audit ------> source manifest, index, and assets
```

The model-history generator's framework order, title map, filters, and skill
slug list must agree. Current evidence will determine whether Hunyuan3,
MOSS-VL, and Qwen3.6 are added per framework; a model is omitted when its
upstream repository has no meaningful implementation history. Existing
TensorRT-LLM and TokenSpeed histories are manually diff-reviewed from their
previous recorded heads because they are curated rather than generated.

Open-PR watch entries remain nonempty, genuinely open at generation time, and
classified using the same model vocabulary as the histories. Review-corpus
generation must remain bounded enough for a full refresh and must report its
collection cutoff and immutable source evidence. Generated output is reviewed
for semantic accuracy after automation completes.

The architecture catalog adds the public Kimi K3 architecture asset from the
audited source repository, records its exact source commit, and updates source
notes. Existing release archives remain historical artifacts rather than being
silently replaced.

### 3. Operational Skills and Serving Configurations

Every `SKILL.md` is checked against the current behavior of its scripts and
references. The refresh specifically:

- removes vLLM's retired `max_num_partial_prefills` and
  `max_long_partial_prefills` flags from serving configurations, validators, and
  examples while retaining supported long-prefill controls;
- adds only current model configurations that have authoritative model IDs,
  framework support, and defensible launch settings;
- aligns model-history lookup slugs with newly generated histories;
- updates SOTA-loop, dossier, profiler, architecture, capacity, incident, and
  benchmark instructions when their actual commands or evidence paths changed;
- keeps prompt copies synchronized when a shared operational model list changes;
- keeps skills concise and moves detailed, changing evidence into references or
  scripts.

New model entries such as Qwen3.6, MiniMax M3, Inkling, Unlimited OCR, Kimi K3,
or DeepSeek V4 are evaluated independently. Their recent existence alone is not
enough to add them to every tool: each destination must be able to represent
the model accurately and validate its configuration.

### 4. Repository Metadata and Automation

Repository-level updates cover:

- README launch examples for current Claude Code model aliases, permission
  modes, and explicit Codex approval/sandbox flags;
- plugin and marketplace metadata, including the missing marketplace
  description;
- current stable CI action majors and pre-commit tool releases;
- link-checker action and binary versions;
- validation instructions and update-checklist dates or source heads.

Dependency updates are accepted only after their configured hooks run
successfully. Pinning continues to follow the repository's existing security
and reproducibility style.

## Error Handling and Integrity Rules

- Reject invalid numeric CLI inputs before issuing requests or dividing values.
- Propagate HTTP status failures with the request context intact.
- Emit valid JSON with no prose contamination whenever JSON mode is selected.
- Represent missing log evidence as unknown rather than fabricating a capacity
  value.
- Fail generated-artifact checks when source heads, model sets, indexes, or
  open-PR state disagree.
- Distinguish merged, open, and closed-unmerged pull requests. Closed-unmerged
  work may be mentioned as an explored approach only when explicitly labeled.
- Resolve all referenced source paths against the recorded upstream checkout.
  A stale path must be updated, removed, or explicitly marked as historical.
- Do not commit caches, temporary clones, credentials, profiling scratch data,
  or machine-specific files.

## Testing and Validation

Validation proceeds in layers:

1. Add focused regression tests and run each new test red, then green.
2. Run the complete `unittest` and `pytest` suites with bytecode/cache output
   disabled where practical.
3. Compile all Python scripts and run shell syntax checks.
4. Run plugin and marketplace validators.
5. Run model-history and open-PR-watch generation in check or temporary-output
   mode before replacing committed data.
6. Validate all model-profile/config artifacts, indexes, source SHAs, paths,
   PR states, duplicate entries, and stale-date markers.
7. Execute representative live-profiler and Nsight Compute workflows, plus the
   repository's isolated MiniMax M3 and remote B200 checks prescribed by the
   update guide. Record exact commands, hardware, and outcomes.
8. Inspect the final diff for generated noise, unsupported claims, secrets,
   temporary artifacts, and unrelated edits.
9. Push the branch, open a draft PR, and verify GitHub checks. Fix in-scope CI
   failures before handoff.

Hardware-dependent checks that cannot produce meaningful evidence on an
available machine are reported precisely; they are never replaced by a claim
that a CPU-only check proved GPU behavior.

## Commit and Pull Request Strategy

Use small, coherent commits:

1. approved design and implementation plan;
2. regression-tested deterministic tool fixes;
3. upstream source and generated-artifact refresh;
4. operational skills, configurations, and architecture updates;
5. repository metadata, dependency, documentation, and CI refresh;
6. validation-driven corrections.

Before each commit, stage only files belonging to that unit. The final draft PR
targets the repository's default branch and explains the motivation, root
causes, user-visible impact, exact source snapshot, and validation evidence.

## Non-Goals

- Redesigning the repository layout or replacing existing CLIs.
- Adding speculative model support without current source and validation.
- Optimizing production CUDA kernels in upstream frameworks.
- Rewriting historical evidence to make closed experiments appear successful.
- Publishing a new release archive or modifying external repositories.

## Acceptance Checklist

- [ ] All audited correctness findings are fixed or explicitly disproved with
      authoritative evidence.
- [ ] All twelve skills match their scripts and current upstream interfaces.
- [ ] Generated histories, open-PR watch, review corpus, profiler evidence, and
      architecture metadata use current immutable heads.
- [ ] Every added model/configuration has a verified source and passes its
      applicable validators.
- [ ] Full local, artifact, remote, profiler, and GPU validation evidence is
      captured.
- [ ] The worktree contains no unrelated or temporary files.
- [ ] A draft pull request is open with green required checks or a documented,
      externally caused pending check.
