---
name: sglang-model-day0-support
description: Build or audit an evidence-driven SGLang Day-0 support program for a new LLM, VLM, MoE, hybrid-attention, or speculative-decoding model. Use when Codex needs to map a model architecture into SGLang runtime work, design a public support PR DAG, create validation and release gates, sanitize private development evidence, distinguish Day-0 requirements from later fixes or optimizations, or review whether an existing model-support PR is release-ready.
---

# SGLang Model Day-0 Support

Turn a model release into a reviewable SGLang support program. Produce evidence,
implementation boundaries, validation gates, and public release artifacts—not
only a launch command.

## Start Here

1. Read [day0-contract.md](references/day0-contract.md) completely.
2. Copy `assets/day0-bundle/` into the active model-support workspace.
3. Lock immutable model, weight, tokenizer/processor, SGLang, dependency, and
   image revisions before describing support.
4. Read [evidence-audit.md](references/evidence-audit.md) before citing PRs.
5. If any input is non-public, read
   [sanitization.md](references/sanitization.md) before writing public output.
6. Read only the relevant case study:
   - [kimi-k3-case-study.md](references/kimi-k3-case-study.md) for hybrid
     KDA/MLA, VLM, DSpark, ReplaySSM, DCP, or dense internal-to-public delivery.
   - [deepseek-v4-case-study.md](references/deepseek-v4-case-study.md) for
     compressed/sparse attention, SWA, mHC, MTP, large rebases, or staged
     cookbook/image/mainline delivery.

Do not implement model code until the scope contract, architecture gap map, PR
DAG, and minimum validation matrix agree.

## Workflow

### 1. Lock the release cut

Fill `scope-contract.md` and `release-lock.md`.

- Record full immutable revisions. Reject floating branches, mutable tags, and
  unpinned object-store paths as release evidence.
- Name required hardware, precision, modality, serving topology, protocol, and
  performance lanes.
- Separate Day-0 commitments from explicitly deferred work.
- Record dependency ownership: upstream package, SGLang patch, bundled wheel,
  image layer, or documented external prerequisite.
- Treat unavailable weights or hardware as an explicit blocked lane, not an
  implicit pass.

### 2. Build the architecture gap map

Compare the public model configuration/reference implementation with the locked
SGLang source. Inspect each capability family in the Day-0 contract.

For each gap, record:

- checkpoint/config field and expected tensor or protocol contract;
- closest SGLang implementation and why reuse is safe or unsafe;
- files/classes/functions expected to change;
- required fallback before any specialized kernel;
- evidence class and validation owner;
- interaction risks with state, graph capture, parallel layout, or protocol.

Use public model-family PR history when available. Do not infer support from a
shared architecture name alone.

### 3. Classify the work

Assign exactly one class:

- `day0-required`: needed to load, serve, produce correct public API output, or
  satisfy an announced release lane.
- `post-day0-fix`: discovered after the release cut and required to restore a
  previously claimed behavior.
- `performance-only`: improves a supported path without defining correctness.
- `experiment-or-revert`: unmerged, reverted, unreachable, or retained only as
  a negative result.

If one patch mixes classes, split it or explain why atomic review is impossible.
Never promote an experiment merely because it exists on a development branch.

### 4. Design the PR DAG

Fill `pr-dag.md`. Prefer reviewable PRs with explicit dependencies:

1. shared public infrastructure;
2. model config, loader, and eager correctness spine;
3. state/cache and required parallel composition;
4. protocol, speculative, and multimodal surfaces;
5. platform-specific kernels or backends;
6. images, cookbook, tests, and release wiring;
7. one umbrella integration PR only when the release branch requires it.

Require each node to name its merge gate, fallback, tests, and public evidence.
Avoid copying an entire development branch when a smaller auditable diff is
possible.

### 5. Build the validation matrix

Fill `validation-matrix.md`. Do not construct the full Cartesian product.

Always include:

- one eager load and deterministic short generation;
- weight mapping and quantization post-processing;
- chat, reasoning, tool, structured-output, streaming-split, and stop-marker
  behavior required by the model;
- state allocation, prefix reuse, eviction, graph padding, and dtype/layout;
- required TP/DP/EP/CP/DCP/PP/PD/EPD roles;
- accuracy, memory capacity, and representative performance;
- one fallback-path test for each specialized backend.

Add risk pairs where two features rewrite the same state, tokens, graph,
collective, or transfer layout. Typical examples are speculative decoding ×
recurrent state, DCP × cache ownership, VLM × PD/EPD, DP attention × MoE
collectives, and CUDA Graphs × alternative streams.

### 6. Execute the seven gates

Execute the gates from the Day-0 contract in order:

1. source;
2. load;
3. protocol;
4. state;
5. topology;
6. quality/performance;
7. release.

For every pass, retain the command, revision, hardware, result, and limitation.
For every failure, record whether it blocks Day-0, narrows the claim, or moves
to the follow-up ledger.

Do not use server startup as evidence for output correctness, state correctness,
topology composition, or performance.

### 7. Synthesize the public PR

Fill `pr-body.md` from reviewed evidence.

- Lead with the support contract and exact public status.
- Explain architecture deltas and implementation boundaries.
- Link public PRs and public source paths.
- Separate required support from optional fast paths.
- State tested hardware, topology, precision, modality, workload, and revision.
- Preserve known limitations and open work.
- For large rebases, include a conflict-decision ledger: upstream behavior,
  model-branch behavior, chosen resolution, owner, and validation.

Run the public evidence collector only for mechanical metadata. Write
motivation, implementation, and limitations manually after reading the diff.

### 8. Track post-Day-0 work

Fill `follow-up-ledger.md`.

- Keep correctness repairs, performance work, platform extensions, and
  experiments/reverts separate.
- Record which original claim each fix changes.
- Reopen a release gate when a fix invalidates its evidence.
- Mark default flips and reverts explicitly; do not rewrite history as if the
  final default was always known.

### 9. Validate and sanitize

Run:

```bash
python3 scripts/validate_day0_bundle.py /path/to/day0-bundle
```

When private inputs exist, create an uncommitted denylist with one forbidden
literal per line:

```bash
python3 scripts/validate_day0_bundle.py \
  /path/to/day0-bundle \
  --denylist /path/to/uncommitted-denylist.txt
```

Collect public PR metadata when needed:

```bash
python3 scripts/collect_public_pr_evidence.py \
  https://github.com/sgl-project/sglang/pull/23882 \
  --output /path/to/public-pr-evidence.json
```

Inspect the generated JSON, then manually read the complete diff and final
mainline code. The JSON is an inventory, not a PR summary.

## Reference Routing

- Read [day0-contract.md](references/day0-contract.md) for every invocation.
- Read [evidence-audit.md](references/evidence-audit.md) before using PR
  evidence or refreshing a case study.
- Read [sanitization.md](references/sanitization.md) whenever private inputs,
  unreleased artifacts, or non-public environments are present.
- Read [kimi-k3-case-study.md](references/kimi-k3-case-study.md) for the public
  Kimi K3 example and its hybrid/VLM failure boundaries.
- Read [deepseek-v4-case-study.md](references/deepseek-v4-case-study.md) for the
  public DeepSeek V4 example and its staged merge/backfill/repair pattern.

## Completion Contract

Finish only when:

- all eight bundle files validate without unresolved markers;
- every required capability has a pass, explicit limitation, or Day-0 blocker;
- every cited PR has a manual diff-reviewed card or a link to one;
- open, closed-unmerged, reverted, and experimental work is labeled correctly;
- public performance claims retain their public source and measurement scope;
- the release lock uses immutable artifacts;
- the sanitization report records the public-evidence and denylist results;
- the public PR body matches the actual release cut.
