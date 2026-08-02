# KDA-Pilot And SGLang Knowledge Routing

## Sources

Build the knowledge pack from clean, pinned Git worktrees:

- KDA-Pilot `diffusion/**`;
- KDA-Pilot's pinned `external/KernelWiki` submodule;
- its pinned `external/ncu-report-skill` submodule;
- its pinned `external/warp-specialization-report-skill` submodule;
- SGLang `python/sglang/multimodal_gen/**` and diffusion-specific docs/tests or
  kernels selected by the builder;
- the reviewed SGLang Diffusion PR/rule audit and history rules from this
  repository when available;
- recent commits touching SGLang's multimodal-generation tree.

The builder records every source commit and file digest. It refuses tracked
changes, missing submodule repositories, and empty source families.

## Routing

Use the lane names from the pinned upstream registry. The builder's current
conservative mapping is:

| Idea | Eligible lane |
| --- | --- |
| exact operator/kernel fusion, compiler, layout-local implementation | `kernel` |
| cache/reuse with the pinned lane's quality contract | `cache` |
| sparse attention or token selection | `pisa` |
| process groups, collectives, sequence/context parallel layout | `topology` |
| sub-16-bit quantization when no compatible lane exists | none; knowledge-only |

Path classification is an index, not proof that an idea belongs in a lane.
The Executor must read the source and obey its appended upstream technique
scope. If an idea conflicts with that scope/correctness mode, do not use it.

Historical GPU thresholds, shapes, speedups, tolerances, and conclusions are
hypothesis seeds only. Remeasure the frozen workload and let Sol decide.

## Injection Procedure

Choose one four-digit sequence for the campaign. Read the selected technique's
`workflow_uid` from the pinned registry. For each selected technique:

1. construct the experiment id exactly as the pinned Master prompt specifies,
   normally `sglang-<workflow_uid>-<sequence>` for model
   `sglang_diffusion`;
2. call upstream `scripts/create_model_experiment.py` with that model,
   workflow id, experiment id, and experiments root;
3. locate the generated `worktree/goals/<workflow_uid>/goal.md` through
   `experiment.json`, not by guessing an external path;
4. run `inject_executor_knowledge.py` with the registered technique;
5. record the resulting goal path, manifest digest, technique, workflow id,
   and experiment id in campaign metadata.

Do this before launching the Master. The injection tool is idempotent by
refusal: a second injection for the same technique fails instead of duplicating
prompt material.

After injection, verify the seed still begins with upstream's generated model
goal. Do not paste or edit the technique scope, loop contract, frozen baseline,
or Master prompt. The pinned `spawn_executor.py` appends those authoritative
sections when it reuses the experiment.

## Executor Evidence

The injected block tells Executors to cite source id, commit, relative path,
and digest for any idea they use. This citation is provenance, not acceptance.
Sol's delivery verifier and Master remain responsible for deciding whether the
candidate is authentic, correct, fast, and visually acceptable.

Knowledge roots and manifests are read-only. Never copy them under `sglang/`,
and never let them enter the final patch.
