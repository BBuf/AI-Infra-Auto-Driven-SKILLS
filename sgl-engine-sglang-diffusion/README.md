# SGL-Engine for SGLang Diffusion

`sgl-engine-sglang-diffusion` is a deterministic evidence controller for a
serial, interactive optimization campaign. The current root agent owns every
hypothesis, code change, benchmark decision, visual review, and integration
decision. The controller locks inputs, runs the authoritative baseline and
profile, issues one work order, verifies submitted evidence, composes verified
candidates, records progress, and packages the final patch.

The controller never starts an AI process. It has no executor agents, Master
agent, AI reviewer, nested Codex or Claude command, or per-agent token budget.
Its `AWAITING_AGENT` state is a deliberate boundary at which the current
conversation reads evidence and continues the campaign.

The verification contract remains compatible with the reviewed Sol-Engine
correctness rules. The Controller also mirrors Sol's six-family search space,
structured candidates, recipes, and technique implementation references.
Locked SGLang, FastVideo, KDA-Pilot, KernelWiki, NCU, warp-specialization,
profiler, or model-history evidence may suggest a hypothesis but cannot weaken
a gate.

## Why the flow is serial

Several technique processes sharing one GPU can contaminate timings, contend
for memory and ports, and turn one slow or blocked lane into a global
integration barrier. This implementation instead permits:

- one interactive root agent;
- one active work order and detached SGLang worktree;
- one candidate GPU measurement at a time;
- one explicit submission boundary per scientific round; and
- integration of the verified latency-positive subset.

Routes are suggestions rather than required parallel lanes. A rejection
returns to `AWAITING_AGENT`; it does not terminate the campaign. Repeated
failure signatures close only the affected hypothesis.

## Search space and knowledge

Deterministic setup creates two bound artifacts before the baseline:

- `SEARCH-SPACE.json` normalizes the locked Sol revision into kernel, cache,
  sparse-attention, quantization, token-pruning, and topology families. It
  preserves method directions, structured candidate requirements,
  model-specific recipes, site documentation, and registered implementations.
- `KNOWLEDGE.json` points to commit-locked, per-file-hashed text snapshots from
  Sol, SGLang, FastVideo, KDA-Pilot, KernelWiki, the NCU report skill, and the
  warp-specialization skill.

KDA's three skill submodules receive independent locks at the exact gitlink
commits stored by the locked KDA revision. They are not treated as files owned
by the parent commit.

The catalog records opportunities, not results. A method moves through
`documented -> referenced -> adapted -> validated`; only the last state has a
passing SGLang end-to-end result.

## Install

Most users should install
[`sglang-diffusion-auto-optimize`](../skills/sglang-diffusion-auto-optimize/)
and let the current conversation operate the controller.

For controller development:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './sgl-engine-sglang-diffusion[dev]'
sgl-diffusion-engine --help
python -m pytest sgl-engine-sglang-diffusion/tests -q
```

## Start a campaign

Provide the machine, model, exact native SGLang Diffusion baseline command,
and measured end-to-end speedup target. The validation prompt file must have
exactly five non-empty prompts.

The skill writes the launch request and invokes:

```bash
sgl-diffusion-engine launch \
  --request /absolute/path/campaign-request.yaml \
  --detach
```

The request has no `agent` command. Launch is idempotent for the same frozen
request. The watchdog may lock sources, run the single authoritative baseline,
and collect the initial profile. It then stops at `AWAITING_AGENT`; it does not
continue in a background AI loop.

For a lower-level start, edit
[`examples/goal.yaml`](examples/goal.yaml) and run:

```bash
sgl-diffusion-engine init --goal examples/goal.yaml --run-root runs/
sgl-diffusion-engine run --campaign runs/<campaign-id>
```

The frozen goal uses schema version 2 and
`execution_mode: interactive_single_agent`.

Resume a recoverable deterministic state with:

```bash
sgl-diffusion-engine resume --campaign runs/<campaign-id>
```

At `AWAITING_AGENT`, use `work` and `claim` instead; `resume` never starts an
AI process.

## Interactive work-order loop

Inspect the current state:

```bash
sgl-diffusion-engine work --campaign runs/<campaign-id> --json
```

At `AWAITING_AGENT`, choose one evidence-backed routed technique:

```bash
sgl-diffusion-engine claim \
  --campaign runs/<campaign-id> \
  --technique kernel
```

`claim` creates exactly one detached worktree and returns:

- `search/<epoch>/AGENT-WORK.json`;
- the absolute worktree and required `DELIVERY.json` path;
- the required `AGENT-REVIEW.json` path;
- hashes of the frozen source, baseline, profile, and technique contract; and
- bound paths and hashes for `KNOWLEDGE.json` and `SEARCH-SPACE.json`; and
- scientific rounds used and remaining.

The current root agent edits the returned worktree, commits the candidate,
runs the complete frozen workload, inspects the diff and evidence, and writes
the same-agent review. Every implementation manifest cites at least one exact
source, commit, path, and raw SHA-256 from a bound knowledge index. It then
submits the exact delivery path:

```bash
sgl-diffusion-engine submit \
  --campaign runs/<campaign-id> \
  --delivery /absolute/worktree/DELIVERY.json
```

`submit` is the scientific-round boundary. The controller binds the delivery
digest, recomputes verification, and either integrates the candidate or
returns actionable findings at `AWAITING_AGENT`. Editing a delivery after
submission fails closed.

If evidence rules out a route:

```bash
sgl-diffusion-engine skip \
  --campaign runs/<campaign-id> \
  --technique quantization \
  --classification unsupported \
  --reason 'NVFP4 requires Blackwell; the locked machine is Hopper'
```

`unsupported` and `no_gain` close a technique. `blocked` records a recoverable
environmental issue. If a closed technique already has a verified candidate in
the current stack, the controller preserves its evidence, excludes it from
selection, advances to a new non-scientific epoch, and remeasures the remaining
verified subset.

## Same-agent review

The current agent is both implementer and reviewer; the flow does not pretend
those roles are independent. `AGENT-REVIEW.json` binds:

- the baseline and candidate commits;
- the exact binary full-index diff SHA-256;
- the lossless method argument or quality-gated activation SHA-256;
- the review decision and findings; and
- the authenticity or five-prompt visual-verdict SHA-256.

The controller verifies those bindings deterministically. Quality-gated
candidates additionally require aligned Sol LPIPS evidence and five reviewed
prompt outputs. External VLM verdict services are not used.

## Scientific rounds and failures

A round is consumed only when a distinct candidate finishes the complete
frozen workload and is explicitly submitted. GPU or port contention,
disconnects, preflight dependency failures, launch failures before model
execution, and malformed metadata without a measured run consume no round.

The search reaches `SEARCH_SPACE_EXHAUSTED` only after every routed suggestion
is explicitly closed or consumes its scientific budget and its complete
family-level Sol coverage requirement was reviewed. A PISA-only sparse search
or a three-family-only cache search is incomplete. A performance plateau is
not a proof of impossibility. `UNREACHABLE_CERTIFIED` requires a
deterministically checkable lower-bound certificate.

## Verification and integration

For every isolated and integrated run, the verifier recomputes:

- baseline and candidate full-workload latency and speedup;
- source, command, workload, output, and review hashes;
- native backend, engagement, and fallback receipts;
- lossless method equivalence; or
- aligned LPIPS plus the five-prompt visual binding.

The correctness branches are deliberately asymmetric:

| Technique | Mode | Required gate |
| --- | --- | --- |
| kernel, topology | lossless | Unchanged global steps and DiT calls, unchanged logical work, method/code review, authentic real media. |
| cache, sparse attention, quantization, token pruning | quality-gated | Full workload, positive engagement, no fallback, aligned LPIPS, and five-prompt same-agent visual review. |

Only verified candidates with measured speedup greater than `1.0x` enter the
integration subset. Integration remeasures the composed stack; isolated
speedups are never added. A conflict removes or revises the conflicting
candidate instead of waiting for every routed technique.

## Progress

```bash
sgl-diffusion-engine progress --campaign runs/<campaign-id>
sgl-diffusion-engine progress --campaign runs/<campaign-id> --json
sgl-diffusion-engine progress --campaign runs/<campaign-id> --watch
```

`--watch` returns when the campaign becomes terminal or yields at
`AWAITING_AGENT`. A typical projection is:

```text
Wan-AI/Wan2.2-T2V-A14B-Diffusers · gpu-host · TARGET 2.00x

performance [██████████████░░░░░░] 1.68x / 2.00x
search      [███████░░░░░░░░░░░░░] 4 / 12 rounds
phase       AWAITING_AGENT · epoch 4 · elapsed 06:14:09
latency     128.4000s baseline -> 76.4286s integrated

technique          state       gate          rounds  isolated e2e
kernel             integrated  passed             2         1.27x
cache              verified    passed             1         1.18x
quantization       unsupported pending            0             -
-------------------------------------------------------------------
integrated stack                                  1.68x
```

Progress reports submitted scientific rounds, current work order, explicit
dispositions, verified isolated measurements, and the remeasured integrated
stack. The CLI reports current-conversation token usage as unavailable; it
does not estimate tokens from text or elapsed time.

## State model

```text
NEW -> BASELINE_LOCKED -> PROFILED -> AWAITING_AGENT
AWAITING_AGENT -> SEARCHING -> INTEGRATING -> FINAL_VERIFYING
FINAL_VERIFYING -> AWAITING_AGENT | TARGET_REACHED
```

Recoverable resource states preserve the durable campaign. The terminal
states are:

| State | Meaning |
| --- | --- |
| `TARGET_REACHED` | The integrated patch reached the target and passed clean-room revalidation. |
| `SEARCH_SPACE_EXHAUSTED` | Every routed technique was reviewed or exhausted without reaching the target. |
| `UNREACHABLE_CERTIFIED` | A checkable lower bound proves the target is outside the scoped achievable region. |

The watchdog advances deterministic setup and verification bursts, but stops
at `AWAITING_AGENT`. `resume` never launches an AI command.

## Artifacts

```text
runs/<campaign-id>/
├── CAMPAIGN.json
├── GOAL.yaml
├── SOURCE-LOCKS.json
├── KNOWLEDGE.json
├── SEARCH-SPACE.json
├── BASELINE-COMMAND.json
├── BASELINE.json
├── ROUTES.json
├── VERIFIED-CANDIDATES.json
├── TECHNIQUE-DISPOSITIONS.json
├── PROGRESS.json
├── state.sqlite
├── events.jsonl
├── knowledge/<source>/<commit>/
│   ├── index.json
│   └── references/
├── profiles/0/PROFILE-DIGEST.json
├── search/<epoch>/
│   ├── AGENT-WORK.json
│   └── worktree/
│       ├── AGENT-REVIEW.json
│       └── DELIVERY.json
├── integration/<epoch>/attempt-*/
└── patch/
    ├── sglang.patch
    ├── manifest.json
    ├── SHA256SUMS
    └── apply_and_verify.sh
```

Legacy multi-agent campaign directories are not resumed by this execution
mode. Start a new schema-v2 campaign so old executor state cannot be silently
interpreted as a root-agent work order.

## Applying a result

The patch targets the exact SGLang commit locked at campaign start:

```bash
cd /path/to/sglang
git checkout <locked-sglang-sha>
/path/to/runs/<campaign-id>/patch/apply_and_verify.sh
```

The script checks `HEAD`, validates and applies `sglang.patch`, runs packaged
CPU checks, and prints the exact GPU revalidation command.

Quantized candidates that require derived weights must include an immutable
URI, revision, byte size, and SHA-256. Credentials stay in the execution
environment and are redacted from receipts. Preserve campaign evidence and
clean only campaign-owned processes and temporary paths.

When the packaged SGLang change exposes an optimization profile, its runtime
entry remains `--agent-optimization off|auto|<profile-id>`. Here “agent”
labels the generated profile namespace; it does not launch another AI process.
