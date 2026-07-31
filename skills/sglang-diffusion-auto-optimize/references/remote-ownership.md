# Remote Campaign Ownership

## Responsibility Boundary

The host skill owns SSH/jumpbox access, container entry, persistent paths, GPU
allocation, process-ownership rules, and host-specific cleanup.

The current interactive root agent owns every hypothesis, code change, visual
or method review, candidate command, and decision to claim or skip a
technique.

The Python tool owns immutable source/workload locks, baseline and profile
artifacts, one exclusive worktree, deterministic verification, state,
selected-subset integration, progress, and patch packaging. It never launches
an AI process.

## Launch Safety

Before launch:

1. inspect GPU processes and memory;
2. verify the SGLang origin and fetched full commit;
3. preserve dirty user work;
4. validate the native benchmark and five-prompt file;
5. choose persistent storage; and
6. inherit credentials through the environment.

The detached watchdog runs deterministic setup and stops at `AWAITING_AGENT`.
The current conversation then performs the work-order loop. If the
conversation is interrupted, resume the same campaign in one root-agent
conversation; do not create a background AI worker.

Use one candidate GPU run at a time. Do not use broad `pkill`, recursive
deletion, `git reset --hard`, or cleanup outside campaign-owned paths.
