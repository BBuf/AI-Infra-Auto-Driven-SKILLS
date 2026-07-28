# SGLang Model Day-0 Support Skill Design

## Purpose

Create a reusable `sglang-model-day0-support` skill that turns a new model
release into an auditable SGLang Day-0 support program and public pull request
bundle.

The skill must do more than produce a launch command. It must identify the
model architecture delta, divide work into reviewable pull requests, define
cross-feature validation gates, lock release artifacts, distinguish Day-0
requirements from later optimization, and prevent private development metadata
from leaking into public artifacts.

Two public case studies ground the workflow:

- Kimi K3, centered on the public
  [`sgl-project/sglang#32541`](https://github.com/sgl-project/sglang/pull/32541)
  Day-0 support pull request and its public follow-ups.
- DeepSeek V4, centered on the merged
  [`sgl-project/sglang#23882`](https://github.com/sgl-project/sglang/pull/23882)
  mainline support pull request, the
  [`#24793`](https://github.com/sgl-project/sglang/pull/24793) backfill, and the
  subsequent public repair and optimization history.

## Success Criteria

The work is complete when:

1. The new skill triggers for SGLang new-model bring-up, Day-0 planning, support
   PR design, validation-matrix design, and existing Day-0 PR audits.
2. `SKILL.md` provides a concise executable workflow and loads detailed
   references only when needed.
3. The skill emits a consistent Day-0 bundle containing a scope contract,
   architecture gap map, PR DAG, validation matrix, release lock, public PR
   body, follow-up ledger, and sanitization report.
4. The Kimi K3 case study explains the public support surface, public
   follow-ups, important failure boundaries, and clean-room publication rules
   without exposing private repository identifiers or environment details.
5. The DeepSeek V4 case study explains the staged cookbook/image/support flow,
   rebase decision ledger, mainline support, immediate backfill, and post-Day-0
   repair loop.
6. Scripts collect only deterministic public PR metadata and validate bundle
   structure. They never generate motivation, technical conclusions, or
   performance claims.
7. Tests prove bundle validation, public-evidence checks, and sanitization
   failures using synthetic fixtures.
8. A worked synthetic demonstration shows how the skill finds gaps and designs
   a Day-0 PR program.
9. Repository validation, skill validation, targeted tests, link checks, and a
   final private-identifier scan pass.
10. The change is published as one focused draft pull request against `main`.

## Approaches Considered

### 1. Core workflow, case-study references, and deterministic audits

Keep the operational workflow in a short `SKILL.md`, move model-specific
knowledge into two references, ship copyable bundle templates, and add small
scripts for public evidence collection and bundle validation.

This is the chosen approach. It preserves context, makes the skill useful for
models beyond Kimi K3 and DeepSeek V4, and keeps human technical judgment
separate from mechanical checks.

### 2. One monolithic skill document

Put the workflow and both model histories into one `SKILL.md`. This is simpler
to package but would load hundreds of PR-derived facts on every invocation and
would be difficult to keep current.

### 3. Generator-first support planning

Generate the support plan directly from repository diffs and PR titles. This
would be fast, but it cannot reliably distinguish Day-0 requirements from
experiments, reverts, post-release repairs, and model-specific performance
work. It also conflicts with the repository's manual PR-diff dossier standard.

## Skill Layout

```text
skills/model-optimization/sglang-model-day0-support/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── assets/
│   └── day0-bundle/
│       ├── architecture-gap-map.md
│       ├── follow-up-ledger.md
│       ├── pr-dag.md
│       ├── pr-body.md
│       ├── release-lock.md
│       ├── sanitization-report.md
│       ├── scope-contract.md
│       └── validation-matrix.md
├── references/
│   ├── day0-contract.md
│   ├── deepseek-v4-case-study.md
│   ├── evidence-audit.md
│   ├── kimi-k3-case-study.md
│   └── sanitization.md
└── scripts/
    ├── collect_public_pr_evidence.py
    └── validate_day0_bundle.py
```

Targeted tests live in
`tests/test_sglang_model_day0_support.py`. A worked synthetic bundle lives in
`docs/assets/sglang-model-day0-support-demo/` so users can inspect the output
without making the skill load it during normal execution.

## Component Design

### `SKILL.md`

The main document is an imperative, stage-gated workflow under 500 lines. It
defines:

1. evidence and release-cut locking;
2. architecture gap analysis;
3. Day-0 scope classification;
4. PR DAG construction;
5. risk-driven validation matrix construction;
6. implementation and release gates;
7. public PR synthesis;
8. post-release repair tracking;
9. final public-evidence and sanitization checks.

It directly links every reference and says when to read it. It does not repeat
case-study details.

### Day-0 contract reference

`references/day0-contract.md` defines the required artifact bundle and a
portable capability taxonomy:

- model configuration, registration, loading, and weight mapping;
- attention, recurrent state, compression state, and cache layouts;
- dense and MoE layers, activation contracts, routing, and quantization;
- speculative decoding and accepted-state commit/rollback;
- text protocol, reasoning, tools, structured output, and stop conditions;
- multimodal preprocessing, encoder execution, and feature transport;
- TP, DP attention, EP, CP/DCP, PP, PD, EPD, and combinations;
- CUDA Graph capture, alternative streams, overlap, and fallback;
- NVIDIA, AMD, NPU, XPU, CPU, packaging, cookbook, and CI coverage.

It defines four evidence classes: `day0-required`, `post-day0-fix`,
`performance-only`, and `experiment-or-revert`.

### Evidence audit reference

`references/evidence-audit.md` requires a manually reviewed evidence card for
every PR cited as technical evidence. Each card records public link, state,
immutable head, diff scope, motivation, implementation, short real excerpt,
reviewed files, validation implications, and limitations.

The reference explains that scripts may collect metadata and file inventories
but may not write motivation, implementation summaries, or conclusions.

### Sanitization reference

`references/sanitization.md` defines a clean-room publication boundary:

- public artifacts cite only public upstream URLs and public source paths;
- private PR numbers, repositories, branch names, commits, authors, machines,
  IPs, filesystem paths, image registries, artifact identifiers, and benchmark
  round identifiers are forbidden;
- the internal-to-public mapping is ephemeral and is never committed;
- a lesson without public corroboration is either expressed as an unattributed
  generic guardrail or omitted;
- open and closed-unmerged PRs are not presented as shipped behavior;
- performance claims require a public source and retain their exact hardware
  and workload boundary.

### Kimi K3 case study

`references/kimi-k3-case-study.md` uses public evidence only. It covers:

- the model spine: hybrid KDA/MLA, latent MoE, MXFP4, VLM, and serving protocol;
- memory and state: MLA KV, KDA recurrent state, radix/HiCache, ReplaySSM;
- serving compositions: DSpark, DCP, DP attention, EP, PP/PD/EPD, CUDA Graphs;
- packaging and hardware: public Docker, cookbook, AMD and NPU extensions;
- the public umbrella PR and public follow-ups;
- failure lessons such as graph-padding sentinels, stream topology, logical
  versus physical cache locations, symmetric-memory allocation, parser marker
  fragmentation, and shape-gated kernels;
- the distinction between public Day-0 support, immediate correctness repairs,
  hardware enablement, and later kernel optimization.

The case study links to the existing bilingual Kimi PR history for detailed
diff-reviewed cards instead of duplicating the full dossier.

### DeepSeek V4 case study

`references/deepseek-v4-case-study.md` covers:

- cookbook and hardware recipes preceding mainline support;
- development-branch Docker artifacts and a reviewable rebase decision ledger;
- model, sparse/compressed attention, SWA, compression-state, mHC, MTP, parser,
  and quantization surfaces;
- the mainline support PR and the immediate missing-commit/test backfill;
- post-Day-0 categories: cache and state correctness, PP/PD/CP/DP composition,
  platform support, quantization, graph capture, speculative decoding,
  performance work, default flips, and reverts;
- why release completion requires a cut-specific gap ledger and follow-up
  ownership, not only a merged model file.

The case study links to the existing bilingual DeepSeek V4 history, whose
public PR cards remain the detailed source of truth.

### Public evidence collector

`scripts/collect_public_pr_evidence.py` accepts public GitHub PR URLs and emits
JSON containing:

- repository and PR number;
- public URL, title, state, draft flag, timestamps, and head SHA;
- additions, deletions, changed-file count, and changed-file inventory.

It rejects non-GitHub URLs and repositories outside an explicit allowlist. It
does not interpret the diff.

### Bundle validator

`scripts/validate_day0_bundle.py` validates a generated bundle directory:

- all required files exist;
- no placeholder markers remain;
- every evidence URL is a public GitHub URL allowed by policy;
- required scope, classification, validation, release-lock, and PR sections
  exist;
- open evidence is labeled open;
- forbidden strings supplied through `--denylist` are absent;
- common secret, private-host, absolute-path, and private-PR patterns are
  absent.

It returns non-zero with actionable findings and supports JSON output for CI.

### Bundle templates

Templates use explicit fill markers that the validator rejects until resolved.
They are copied into the user's active model-support workspace, not edited
inside the installed skill.

The templates encode:

- immutable artifact locks;
- architecture capability status and owner;
- PR dependency edges and merge gates;
- risk-pair validation rather than an unbounded feature cross-product;
- public PR narrative and validation evidence;
- known gaps and post-Day-0 owners.

## Data Flow

```text
public model/config/docs + target SGLang source SHA
                         |
                         v
              architecture gap map
                         |
       public PR history + manual diff review
                         |
                         v
        Day-0 classification and evidence ledger
                         |
                         v
             PR DAG + validation matrix
                         |
                         v
      implementation gates + release artifact lock
                         |
                         v
             sanitized public PR bundle
                         |
                         v
            post-Day-0 repair/optimization ledger
```

Private development evidence may inform the human's reasoning before the flow,
but it cannot enter the committed inputs or outputs. The publication boundary
starts at the evidence ledger.

## Day-0 Gates

The skill defines seven ordered gates:

1. **Source gate:** immutable model/config/weight and SGLang revisions.
2. **Load gate:** configuration detection, weight mapping, quantization
   post-processing, and one short deterministic generation.
3. **Protocol gate:** chat template, reasoning, tools, structured output,
   streaming fragmentation, stop markers, and invalid-input behavior.
4. **State gate:** KV/recurrent/compression state allocation, prefix reuse,
   eviction, commit/rollback, graph padding, and dtype/layout invariants.
5. **Topology gate:** required parallel and disaggregated roles plus selected
   high-risk feature intersections.
6. **Quality/performance gate:** accuracy in band, baseline throughput/latency,
   memory capacity, and proof that intended fast paths engage.
7. **Release gate:** public source, images, cookbook, CI, artifact locks,
   limitations, and follow-up ownership.

A gate cannot be marked complete solely from a successful server start.

## Error Handling

The workflow fails closed on:

- floating revisions or mutable artifact tags;
- inaccessible or non-public evidence URLs;
- missing required bundle files or unresolved placeholders;
- private identifiers or unapproved repositories in public artifacts;
- performance claims without a public source and workload boundary;
- open or reverted code presented as shipped support;
- unsupported feature combinations lacking either a test or an explicit
  documented exclusion.

The collector reports network and GitHub API failures without emitting partial
success. The validator reports all independent findings in one run so a user
can correct the bundle without repeated single-error cycles.

## Testing Strategy

`tests/test_sglang_model_day0_support.py` uses temporary synthetic bundles and
mocked public PR payloads. It verifies:

- complete bundles pass;
- missing artifacts and unresolved template markers fail;
- public `sgl-project/sglang` PR URLs pass;
- private or disallowed repository URLs fail;
- explicit denylist entries, absolute paths, host/IP markers, and secret-like
  tokens fail;
- open PR evidence without an `open` limitation fails;
- collector input parsing and output schema are deterministic;
- scripts never synthesize motivation or implementation fields.

Repository validation includes:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/model-optimization/sglang-model-day0-support
pytest -q tests/test_sglang_model_day0_support.py
pre-commit run --all-files
python3 skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py \
  docs/assets/sglang-model-day0-support-demo
```

A repository-wide staged-diff scan checks for private identifiers and
unsupported GitHub origins before publication.

## Demonstration

The worked demo models a fictional hybrid MoE VLM with:

- alternating full and recurrent attention;
- a quantized routed MoE;
- reasoning and tool calling;
- speculative decoding;
- multimodal preprocessing;
- TP/EP plus PD as required release topologies.

The demo shows two skill modes:

1. **Plan mode:** produce the architecture gap map, PR DAG, risk-pair validation
   matrix, release lock, and public PR body.
2. **Audit mode:** detect missing streaming parser tests, remote draft loading,
   recurrent-state transfer, graph-padding coverage, image packaging, and
   follow-up ownership.

The demonstration contains no unreleased model or environment data.

## Pull Request Strategy

Use one branch, `agent/add-sglang-model-day0-support`, based on `origin/main`.
Keep commits focused:

1. approved design;
2. implementation plan;
3. skill scaffold and core references;
4. case studies and templates;
5. scripts, tests, demonstration, and validation fixes.

Open one draft pull request against `BBuf/AI-Infra-Auto-Driven-SKILLS:main`.
The PR body lists the public evidence boundary, generated artifacts, validation
commands, and demonstration result.
