# SGLang Diffusion Model Adapter

## Principle

The adapter makes an exact native SGLang Diffusion workload satisfy the model,
candidate, runtime, and artifact interfaces of the pinned Sol revision. It is
mechanical glue. Upstream Sol owns all decisions made from those artifacts.

Derive the actual TOML fields and output schemas from the closest examples at
the pinned Sol commit. Do not blindly copy a design-time example when the
upstream schema changed.

## Model Contract

Use the model id `sglang_diffusion`. Its model contract should:

- point `baseline.manifest` at the baseline candidate;
- point `baseline.runtime_root` at the adapter runtime;
- point `baseline.eval_profile` at the appropriate pinned Sol evaluation
  profile;
- include only the adapter, candidate schema, pinned upstream launch/collect
  support, and evaluation closure required by the current contract;
- use `[[baseline.external_copy]]` to copy the frozen SGLang checkout to
  `sglang/`;
- explicitly exclude `.git`, `.git/**`, `**/.git`, and `**/.git/**` because a
  linked worktree uses a `.git` file; also exclude caches, build products,
  runs, outputs, checkpoints, and credentials;
- reference model weights, Python environments, datasets, and large prompt
  assets without copying them when upstream supports `reference_only`.

The flat `models/sglang_diffusion.toml` profile must bind the model id, runtime
script, exact official workload configuration, environment, resource envelope,
recorded baseline, and default techniques using the current upstream schema.
Do not encode a custom quality threshold or scheduler policy there.

## Frozen Workload

Translate the caller's command without changing its semantics. Freeze:

- SGLang base commit and Python/import path;
- model/checkpoint and revision;
- prompt/input set and prompt count;
- seed, resolution, frame/audio length, steps, guidance, batch size;
- TP/PP/CP/SP/Ulysses/CFG/world-size topology;
- dtype, quantization state, compile and cache flags already present;
- GPU visibility and hardware identity;
- warmup policy and timed region;
- output and evaluation artifact locations.

Secrets remain inherited environment variables and never enter TOML, logs,
knowledge, or patches.

Run one authoritative baseline with the pinned upstream candidate launcher.
The adapter must produce every canonical artifact required by the pinned
collector/verifier, including a positive total duration and identical timing
scope for baseline and candidates. Preserve native SGLang logs and metrics as
additional evidence; do not replace Sol's canonical fields.

## Patch-Backed Candidate Activation

Every source-changing candidate must be reproducible as an SGLang patch
against the frozen main commit. Use this runtime protocol:

1. An Executor edits only its experiment-local `sglang/` source copy.
2. It exports a component patch against the frozen SGLang commit and records
   the patch digest in its candidate/run provenance.
3. The candidate manifest activates an ordered list of SGLang patches plus any
   non-source runtime switches accepted by the pinned candidate schema.
4. For every run, the adapter materializes a fresh run-owned SGLang source tree
   from the frozen base, applies that exact ordered patch list with
   `git apply --check`, builds/installs as required, and executes the frozen
   command from that tree.
5. The run metadata records the base commit, ordered patch paths/digests,
   resulting source-tree path, command, environment allowlist, and build
   provenance.

This makes Sol's normal recipe composition operate on explicit activations.
The upstream Master decides which component activations are compatible and
which integrated point passes. The adapter does not resolve conflicts or
post-compose an unmeasured result.

## Integrated Source Binding

The accepted integrated run must retain its run-owned complete SGLang source
tree until final delivery. Its source metadata must be content-bound to the
same candidate/run referenced by the accepted integrated frontier point.

Patch export is forbidden from:

- a standalone Executor worktree;
- a best-looking but unaccepted run;
- a manually reconstructed combination after Sol finishes;
- a tree whose patch list or base commit differs from the measured integrated
  run.

## Preflight

Before the authoritative baseline:

1. run the pinned model materializer in dry-run mode;
2. run the pinned candidate launcher help/dry-run path;
3. verify imports, paths, prompt/input count, and output permissions;
4. verify a no-op patch activation leaves the frozen command unchanged;
5. verify canonical benchmark and media/frame artifacts can be collected;
6. verify the source metadata points to the run-owned SGLang tree.

Do not consume the one frozen baseline as an informal preflight.
