# Model PR Optimization History

This directory stores PR-driven model optimization histories by serving
framework. These records are documentation, not installable per-model skills:
the directory acts as one queryable knowledge base for model-family PR evidence.

- `sglang/`: SGLang model histories and audits.
- `vllm/`: vLLM model histories and audits.
- `tensorrt_llm/`: TensorRT-LLM model histories and audits.
- `tokenspeed/`: TokenSpeed model histories and audits.
- `SKILL.md`: agent instructions for using this directory as knowledge.
- `scripts/query.py`: small local search helper for model slugs, doc paths, and
  keyword snippets.
- `open-pr-watch.md`: generated watch list for current open PRs that may affect
  benchmark, profiler, and model-history guidance.

Each model history is bilingual when practical (`README.zh.md` and
`README.en.md`) and should be grounded in inspected PR diffs, source files, and
validation/risk notes. SGLang, vLLM, TensorRT-LLM, and TokenSpeed entries use
the same timeline plus per-PR diff-card format whenever a model family has
upstream PR evidence.

Model availability is framework-specific rather than a cross-framework
promise. For example, MOSS-VL is SGLang-only and Inkling has no TensorRT-LLM
implementation at the audited heads.

All four framework trees are rebuilt by
`tools/rebuild_model_pr_history_from_git.py` from upstream git history. Merged
PRs that already have an audited card keep that card verbatim; only new or
still-open PRs are fetched again. Hand-written dated notes (a source-head
refresh, a backfill audit, or a reviewed kernel addendum) sit between the title
and `Implementation File Coverage` and survive regeneration. Read them first.

Open PRs are deliberately kept out of the merged-history cards until their
diffs are manually reviewed. Regenerate `open-pr-watch.md` before long refresh
work and treat it as a triage queue, not as a source of implemented behavior.

Generated entries labeled **not a manual audit** are PR discovery inventories.
Read their full diffs and current callers before making optimization conclusions;
API file counts and snippets do not establish complete diff coverage. Existing
manual audit notes retain their original dates.

Quick queries:

```bash
python3 scripts/query.py --list
python3 scripts/query.py --framework sglang --model qwen3-core --paths-only
python3 scripts/query.py --framework vllm "qwen3 fused qk norm"
```

Optimization work should read the matching target-framework history before
patch planning, read competitor history when vLLM, TensorRT-LLM, or TokenSpeed
is the leading competitor, and save the short extracted evidence under
`history/model-pr-history-notes.md`.
