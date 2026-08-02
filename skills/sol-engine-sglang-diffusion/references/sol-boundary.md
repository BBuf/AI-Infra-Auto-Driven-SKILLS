# Upstream Sol Engine Boundary

## Authority

Always fetch and pin the current `sol-engine` branch from
`https://github.com/NVlabs/Sana.git` at campaign launch. The branch commit—not
this reference—is the executable specification.

At the source lock used to design this skill, the full entrypoint is
`orchestration/run_orchestrated_experiment.py`. It freezes one baseline,
launches one Master, and provides a heartbeat watchdog. The Master owns
Executor scheduling, polling, independent verification, correction/resume,
and integration. Reconfirm these facts at the campaign's pinned commit.

## Allowed Overlay

The campaign-local Sol checkout may add or edit only model-facing integration
files needed for the `sglang_diffusion` model:

```text
models/sglang_diffusion/model.toml
models/sglang_diffusion.toml
models/sglang_diffusion/baseline/**
candidates/sglang_diffusion*.toml
evals/profiles/*sglang*diffusion*.toml
```

Generated campaign output under upstream-declared output/run directories is
expected and is not source overlay.

## Forbidden Source Changes

Do not edit these authoritative paths:

```text
orchestration/**
workflow/**/nodes/codex_executor/**
search/plan_eval.py
tools/vision/**
tools/symposium/**
scripts/create_model_experiment.py
scripts/launch_candidate.py
scripts/collect_run.py
evals/rubrics/**
orchestration/techniques.toml
```

Do not copy their logic into the skill or add a wrapper that makes competing
scheduling, verification, quality, integration, retry, or termination
decisions.

## Required Compatibility Audit

Before building the overlay, inspect the pinned code rather than assuming the
design-time source layout. Confirm:

1. the full runner and Master prompt still have the same authority split;
2. the technique registry supplies the exact selectable lane names,
   `workflow_uid` values, scopes, and correctness modes;
3. custom model IDs resolve from both `models/<id>/model.toml` and
   `models/<id>.toml`;
4. the model contract still supports a frozen external source copy;
5. `create_model_experiment.py` creates an experiment-local seed `goal.md`;
6. `spawn_executor.py` safely reuses a matching pre-created experiment and
   composes its seed with the upstream scope/contract/baseline;
7. the integrated delivery still identifies the accepted point and run.

If any item changed, adapt the model overlay or knowledge-injection procedure
to the new public extension point. Never repair compatibility by editing a
forbidden path.

## Diff Audit

Run a source diff from the pinned Sol commit before dry-run and again before
delivery. Every changed source path must match the allowed overlay. Preserve
the diff/path list with `SOURCE-LOCKS.json` so another reviewer can prove that
Sol strategy and quality code were untouched.

Knowledge injection changes generated experiment seed goals only. Those goals
must be created through upstream Sol's materializer and live under its
campaign output tree; they are not changes to the Master, technique scope, or
loop-contract source.
