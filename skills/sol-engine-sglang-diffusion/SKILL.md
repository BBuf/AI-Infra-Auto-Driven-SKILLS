---
name: sol-engine-sglang-diffusion
description: Run the complete upstream NVlabs Sol Engine optimization flow on an SGLang Diffusion workload while adding pinned KDA-Pilot and SGLang Diffusion implementation knowledge through existing Sol Executor seed goals, then return an apply-checked patch against the SGLang main commit frozen at launch. Use when optimizing an SGLang Diffusion model with Sol's unchanged Master/Executor strategy, correctness modes, quality gates, verification, integration, recovery, and termination rather than a custom controller.
---

# Sol Engine SGLang Diffusion

## Contract

Use upstream Sol Engine as the only optimization engine. This skill is a thin
model/knowledge/output adapter, not an outer controller and not a fork of Sol.

Keep these upstream Sol behaviors authoritative and unchanged:

- Master and Executor orchestration;
- technique registry, scopes, ordering, and round budgets;
- baseline freezing and performance calculation;
- correctness modes and independent verification;
- visual/quality judgment;
- candidate integration, recovery, and termination.

Add only what is necessary to present an SGLang Diffusion workload as a Sol
model, route immutable KDA-Pilot/SGLang knowledge into existing Executor seed
goals, and export the accepted integrated SGLang tree as `sglang.patch`.

## Required Inputs

Obtain or infer:

- target machine or SSH alias;
- model/checkpoint;
- exact native SGLang Diffusion baseline command;
- workload inputs and output/evaluation locations;
- requested Sol techniques or target, when supplied.

Use the closest safe defaults when the user omits a non-critical value. Do not
ask the user to write Sol model files, campaign YAML, knowledge manifests, or
patch commands.

## Read Before Acting

Read all four references:

- [references/sol-boundary.md](references/sol-boundary.md)
- [references/sglang-adapter.md](references/sglang-adapter.md)
- [references/knowledge-routing.md](references/knowledge-routing.md)
- [references/patch-contract.md](references/patch-contract.md)

Read the installed skill whose name exactly matches the target machine and
follow its SSH, container, GPU-allocation, repository, and cleanup rules.

Use `sglang-diffusion-benchmark-profile` and
`sglang-diffusion-performance` when installed to understand the native
workload. Use `model-pr-history-knowledge`, `kernel-knowledge`, KernelWiki, or
NCU skills only as hypothesis/evidence sources. None may override Sol gates.

## Phase 1: Freeze Source Locks

On the target machine, make campaign-owned clean checkouts. Never reset, clean,
or overwrite a user's working tree.

Fetch and resolve full commits for:

1. `https://github.com/NVlabs/Sana.git`, branch `sol-engine`;
2. the SGLang repository, branch `main`;
3. KDA-Pilot and its KernelWiki, NCU-report, and warp-specialization
   submodules;
"Patch against latest SGLang main" means the latest fetched `main` commit at
campaign launch. Freeze it for the entire baseline/candidate campaign. Record
all URLs and commits in campaign-owned `SOURCE-LOCKS.json`.

Initialize KDA-Pilot submodules recursively and verify their checked-out commits
match the parent gitlinks. Fail closed on a dirty or incomplete knowledge
source.

## Phase 2: Audit The Pinned Sol Contract

From the pinned Sol checkout, read:

- `orchestration/README.md`;
- `orchestration/run_orchestrated_experiment.py` and its `--help`;
- `orchestration/techniques.toml`;
- `orchestration/prompts/master.md`;
- `orchestration/bin/spawn_executor.py`;
- `scripts/create_model_experiment.py`;
- the closest current model contract, flat profile, candidate, runtime, and
  evaluation examples.

Confirm that the pinned revision still supports custom model contracts,
experiment-local seed goals, and the full orchestrated runner. If upstream
interfaces changed, adapt only model-facing overlay files and this skill's
commands. Do not patch Sol orchestration or weaken validation to recover old
assumptions.

Follow [references/sol-boundary.md](references/sol-boundary.md) and record an
overlay diff audit before launch.

## Phase 3: Build The SGLang Model Overlay

Create a campaign-local branch or checkout at the pinned Sol commit. Add the
model-facing files described in
[references/sglang-adapter.md](references/sglang-adapter.md), normally:

```text
models/sglang_diffusion/model.toml
models/sglang_diffusion.toml
candidates/sglang_diffusion_baseline.toml
models/sglang_diffusion/baseline/**
evals/profiles/<sglang-diffusion-profile>.toml
```

Use Sol's `baseline.external_copy` contract to copy the frozen clean SGLang
tree into each experiment at `sglang/`. The runtime adapter must execute the
caller's exact workload and emit the artifact schema required by the pinned
Sol revision. It translates artifacts and activates source patches; it does
not judge them.

Run the native SGLang baseline once through upstream Sol's candidate launcher,
then record or pass that canonical run directory exactly as the pinned Sol
runner requires. Freeze the command, prompt/input set, seed, dimensions,
steps, topology, environment, timing scope, and GPU identity.

## Phase 4: Build And Inject Knowledge

Build the immutable manifest in the campaign root:

```bash
python <skill-dir>/scripts/build_knowledge_pack.py \
  --kda-root <frozen-kda-worktree> \
  --sglang-root <frozen-sglang-main-worktree> \
  --output-dir <campaign-root>/knowledge
```

Select techniques only from the pinned
`orchestration/techniques.toml`. Do not invent a new lane. In particular, do
not place sub-16-bit quantization in a lossless kernel lane. If the pinned Sol
revision has no compatible technique, retain that material as knowledge-only.

Before starting the Master, pre-create exactly the experiment IDs that its
pinned prompt will use, through Sol's own `scripts/create_model_experiment.py`.
Then inject only the matching lane into each generated seed `goal.md`:

```bash
python <skill-dir>/scripts/inject_executor_knowledge.py \
  --manifest <campaign-root>/knowledge/KNOWLEDGE-MANIFEST.json \
  --technique <registered-technique> \
  --goal <sol-experiment-worktree>/goals/<workflow-uid>/goal.md
```

Verify in the pinned `spawn_executor.py` that an existing matching experiment
is reused and its seed goal is composed with the unmodified upstream scope,
loop contract, and frozen baseline. If that extension point no longer exists,
stop rather than editing the engine.

## Phase 5: Run Full Upstream Sol Engine

First run the pinned runner's dry-run. Check the model id, sequence, selected
registered techniques, baseline path, experiment IDs, and output paths.

Then launch the complete upstream entrypoint—not individual Executor scripts
and not the legacy controller:

```bash
python orchestration/run_orchestrated_experiment.py \
  --model sglang_diffusion \
  --seq <four-digit-sequence> \
  --baseline-run-dir <canonical-baseline-run-dir> \
  --techs <comma-separated-registered-techniques>
```

Omit optional flags when the pinned profile supplies them. Follow the pinned
`--help`; do not assume an older CLI is stable.

Let Sol's Master spawn, poll, verify, resume, and integrate. Monitor its native
state and report its measured baseline, verified frontier, quality/correctness
results, and terminal state without reinterpreting them. Never accept an
Executor result independently or post-compose candidates outside the Master.

## Phase 6: Export The Accepted Patch

Wait for upstream Sol's integrated delivery. Identify the exact run-owned
SGLang source tree used by its accepted integrated point, as required by the
model adapter. Do not substitute an unmeasured Executor worktree.

Export and apply-check the patch:

```bash
python <skill-dir>/scripts/extract_sglang_patch.py \
  --base-repo <frozen-sglang-git-worktree> \
  --base-commit <frozen-sglang-main-commit> \
  --candidate-tree <accepted-integrated-sglang-source-tree> \
  --output <campaign-root>/delivery/sglang.patch
```

Follow [references/patch-contract.md](references/patch-contract.md). Return the
patch plus the frozen SGLang commit, Sol commit, knowledge-manifest digest, and
the upstream integrated-delivery/evidence paths. If no accepted integrated
source tree exists, report the upstream Sol outcome and withhold a fabricated
patch.

## Completion Rules

Completion is whatever the pinned upstream Sol campaign reports after its own
verification and integration. This skill adds no alternative quality bar,
speedup arithmetic, retry policy, search-exhaustion rule, or unreachable
certificate.

Before handoff verify:

- the Sol checkout differs from its frozen commit only in allowed model-facing
  overlay paths;
- the accepted source tree is bound to the integrated measured run;
- `sglang.patch` applies to the exact frozen SGLang commit;
- the patch contains no Sol files, knowledge files, run artifacts, secrets, or
  machine-local paths.
