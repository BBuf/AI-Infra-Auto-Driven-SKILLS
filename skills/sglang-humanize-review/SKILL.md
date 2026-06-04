---
name: sglang-humanize-review
description: "Perform SGLang code review in the style of human maintainers by consulting the full non-agent PR review episode corpus from project start through 2025-12-31, including inline review threads, top-level PR comments, review submissions, original multilingual text, and multi-round discussions. Use when reviewing SGLang PRs, diffs, patches, or local changes for correctness, tests, performance, GPU/runtime risks, API compatibility, and maintainability."
---

# SGLang Humanize Review

## Overview

Use this skill when the user asks for a human-style SGLang code review or wants
review feedback that resembles SGLang maintainers instead of generic linting.

The bundled corpus is collected from `sgl-project/sglang` PRs from the first
public PR through 2025-12-31, excluding PRs authored by bots or obvious
coding-agent accounts. It is organized as review episodes, not just individual
comments:

- `inline_review_thread`: file/path-specific GitHub pull-review comments with
  `diff_hunk` context and replies grouped by thread.
- `pr_conversation`: top-level PR conversation comments, including design
  discussion, requested repros, benchmark negotiation, and author follow-ups.
- `review_submission`: review summary bodies such as COMMENT and
  REQUEST_CHANGES, preserving the review state.

Every episode preserves PR metadata, reviewer identity, original comment text,
original comment language, timestamps, categories, and multi-round replies when
GitHub exposes them. Read [references/corpus-summary.md](references/corpus-summary.md)
first for coverage, counts, top paths, and category distribution. Do not load
the gzip corpus directly into context; query it with the helper script.

## Corpus Tools

Search the corpus by topic, path, category, or reviewer:

```bash
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py \
  --query cuda --limit 5

python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py \
  --path python/sglang/srt --category correctness --limit 8

python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py \
  --query server_args --format jsonl --limit 3

python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py \
  --kind pr_conversation --query benchmark --limit 5

python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py \
  --kind review_submission --query "request changes" --limit 5
```

The full corpus is:

```text
references/sglang-review-corpus-2024-2025.jsonl.gz
```

Regenerate it only when the user asks to refresh the evidence:

```bash
python3 skills/sglang-humanize-review/scripts/collect_sglang_review_corpus.py \
  --repo sgl-project/sglang \
  --from-beginning \
  --end-year 2025 \
  --out-dir skills/sglang-humanize-review/references
```

## Review Workflow

1. Inspect the actual diff first.
   - Use `git diff`, `gh pr diff`, or the patch supplied by the user.
   - Identify changed SGLang subsystems: server args, scheduler, memory/cache,
     model runner, attention backend, quantization, kernels, OpenAI API,
     metrics, docs, or tests.
2. Read `references/corpus-summary.md`.
   - Note top review surfaces and categories that overlap with the diff.
   - Check episode coverage. Inline evidence is best for file-local findings;
     PR conversation evidence is best for design, benchmarks, repros, and
     author follow-up; review submissions are best for blocking review tone and
     maintainer-level summaries.
   - Preserve the original language of any relevant corpus examples; do not
     translate user-facing comments unless the user asks.
3. Query similar review threads.
   - Search by path first for touched SGLang modules.
   - Search by risk keyword next, for example `cuda`, `kv cache`,
     `server_args`, `openai`, `logprob`, `tp`, `dp`, `eagle`, `fp8`,
     `benchmark`, or `pytest`.
   - Query at least one non-inline source (`--kind pr_conversation` or
     `--kind review_submission`) when the PR changes behavior, tests, docs,
     benchmarking, deployment defaults, or model support.
   - Prefer evidence from the same subsystem over broad keyword matches.
4. Add cross-skill evidence when the diff touches an area covered elsewhere in
   this repository.
   - Model-family implementation or optimization: query
     `model-pr-optimization-history` for the model slug before judging whether
     the change repeats or conflicts with prior PRs.
   - Performance claims or hot paths: use `llm-torch-profiler-analysis`,
     `llm-pipeline-analysis`, or `model-compute-simulation` evidence rather
     than asking for generic "benchmarks".
   - Memory/KV/cache capacity changes: use `llm-serving-capacity-planner`
     expectations for startup logs and capacity accounting.
   - Serving incidents, hangs, or distributed regressions: use
     `sglang-prod-incident-triage` style replay requirements.
5. Produce a code-review response.
   - Lead with concrete findings ordered by severity.
   - Include file and line references from the reviewed diff.
   - Explain the failure mode, not just the preferred style.
   - Suggest a fix or validation step when the issue is actionable.
   - Keep nits separate from correctness, performance, or compatibility risks.
6. If no issue is found, say so clearly.
   - Mention the main residual risk and the test or benchmark coverage that
     would increase confidence.

## SGLang Review Heuristics From The Corpus

Prioritize these risks because they recur heavily in the 2024-2025 human review
threads:

- **Model and quantization behavior**: model config drift, tokenizer assumptions,
  FP8/INT4 quantization paths, MoE routing, speculative decoding, and attention
  backend compatibility.
- **Correctness before style**: edge cases, failed assertions, unexpected error
  codes, shape/dtype mismatches, state cleanup, and silent behavior changes.
- **GPU and kernel paths**: CUDA graph capture, Triton/CUDA kernels, FlashInfer
  and FlashAttention behavior, launch conditions, SM compatibility, and fallback
  behavior.
- **Server API compatibility**: OpenAI-compatible request/response shapes,
  `server_args`, CLI defaults, endpoint behavior, streaming, and backward
  compatibility.
- **Memory and cache lifecycle**: KV cache accounting, radix cache resets,
  memory pool ownership, eviction, fragmentation, and OOM behavior.
- **Distributed runtime**: TP/DP/PP/EP rank assumptions, NCCL paths,
  synchronization, worker state, race conditions, and hang risk.
- **Tests and benchmarks**: ask for targeted tests when behavior changes, and
  ask for benchmark evidence with workload, model, hardware, precision,
  framework commit, and before/after commands when a change claims performance
  or touches a hot path.
- **Docs and examples**: keep docs aligned with CLI defaults, endpoint names,
  model support, install steps, and version-specific behavior.
- **Observability**: review metrics, logs, warning levels, traceability, and
  error messages when operational behavior changes.

## Review Style

Mirror human SGLang review habits:

- Be terse but specific.
- Prefer a question when intent is ambiguous.
- Call out production-facing behavior changes explicitly.
- Do not invent a corpus precedent; query the corpus when using it as evidence.
- Use corpus examples for reviewer instincts and risk surfaces, not as a
  replacement for reading the current diff.
- Keep multilingual comments intact. If a relevant thread is Chinese or another
  language, use it as-is for evidence and answer in the user's language unless
  the user asks otherwise.
- Avoid cargo-culting old comments. Use corpus examples to sharpen the current
  review, not to force the current patch into an old template.

## Output Contract

For a normal review, return:

- Findings first, ordered by severity, with file/line references.
- Open questions or assumptions.
- Test or benchmark gaps.
- A short summary only after findings.

For a review-prep pass before the user opens a PR, return:

- likely reviewer concerns
- missing tests or benchmark evidence
- suggested patch cleanup
- corpus queries used
- cross-skill evidence used, when applicable

For a corpus-backed explanation, include the query terms and summarize the
matched review behavior without dumping long comment bodies.
