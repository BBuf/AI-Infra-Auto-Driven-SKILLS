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

Each model history is bilingual when practical (`README.zh.md` and
`README.en.md`) and should be grounded in inspected PR diffs, source files, and
validation/risk notes. SGLang, vLLM, TensorRT-LLM, and TokenSpeed entries use
the same timeline plus per-PR diff-card format whenever a model family has
upstream PR evidence.

When a doc is rechecked for timeliness, a dated `## <YYYY-MM-DD> PR Backfill
Audit` section is prepended right after the title. It lists PR-numbered merges
that touched the tracked implementation files after the doc's previous freshness
cutoff and are not yet folded into the timeline / diff-audit cards below. Read
that section first to see what changed most recently before trusting the older
cards.

Quick queries:

```bash
python3 scripts/query.py --list
python3 scripts/query.py --framework sglang --model qwen3-core --paths-only
python3 scripts/query.py --framework vllm "qwen3 fused qk norm"
```

SGLang SOTA and Humanize loops should read the matching SGLang history before
patch planning, read competitor history when vLLM, TensorRT-LLM, or TokenSpeed
is the leading competitor, and save the short extracted evidence under
`history/model-pr-history-notes.md`.
