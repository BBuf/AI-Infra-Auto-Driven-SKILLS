# KernelPilot

KernelPilot is a small, stdlib-only framework for Codex-driven kernel
optimization. It keeps the useful parts of the paper-inspired loop while using
our own naming and repo layout.

The core workflow is:

1. define one kernel task with semantic spec, forbidden source files,
   correctness gates, benchmark cases, and target speedup;
2. generate a focused Codex prompt from task metadata, source references, and
   lineage;
3. implement one candidate at a time from the semantic spec;
4. run correctness first, then benchmark every configured case;
5. record both wins and failed directions so the next iteration has context.

## Layout

- `kernelpilot/`: task config loader, prompt builder, lineage store, scoring,
  knowledge-base rendering, and stagnation hints.
- `references/`: method notes plus the kernel-source catalog used as reference
  map before edits.
- `templates/`: reusable prompts, including the Codex goal prompt template.

Task directories and run artifacts are intentionally user-created inputs. Keep
operator-specific reproductions, lineage, and benchmark results outside the
framework tree unless they are meant to be reviewed as examples.

## Contract

KernelPilot intentionally separates semantics from existing optimized kernel
implementations.

A candidate starts from:

- the operator semantic spec in the task file;
- a trusted reference expression or test oracle;
- benchmark shape and dtype contracts;
- previous lineage entries, if any;
- approved reference material such as FlashInfer, FlashAttention,
  Tencent `hpc-ops`, DeepGEMM, and local GPU-kernel references.

A candidate must not copy or pattern-match from the optimized baseline kernel or the
current target implementation. Target-project tests and wrappers may be used as
correctness or integration oracles after the candidate exists.

## Knowledge Base

Print the relevant reference map before editing:

```bash
python -m kernelpilot.cli refs --lane "CUDA C++ JIT"
```

The current catalog includes:

- user-configured local GPU-kernel reference docs;
- user-configured target kernel project worktree;
- verified public kernel repositories: FlashInfer, FlashAttention,
  Tencent `hpc-ops` / hi-ops, and DeepGEMM;
- target-project source is treated as an integration and validation oracle, not
  as an external kernel reference.

## Minimal Use

Validate a task set:

```bash
python -m kernelpilot.cli validate \
  --taskset <taskset.json>
```

Generate the next iteration prompt for a task:

```bash
python -m kernelpilot.cli prompt \
  --taskset <taskset.json> \
  --task <task-id>
```

Compare a candidate benchmark JSON against the matching baseline:

```bash
python -m kernelpilot.cli compare \
  --taskset <taskset.json> \
  --task <task-id> \
  --result <candidate-result.json>
```

Expected candidate result format:

```json
{
  "correct": true,
  "measurements": [
    {"case_id": "n=4194304", "latency_us": 10.9}
  ]
}
```

## Goal Prompt Template

Use [templates/kernel-goal-prompt.md](templates/kernel-goal-prompt.md)
as the top-level Codex goal. Fill in the operator, target speedup, GPU, baseline,
and validation requirements; Codex can then keep iterating until the gate is met.
