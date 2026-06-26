---
name: model-pr-history-knowledge
description: Use when an SGLang, vLLM, TensorRT-LLM, or TokenSpeed serving/model optimization task needs prior model-family PR evidence. Query and read the PR-driven history docs under model-pr-optimization-history before choosing source paths, fast paths, kernel/fusion ideas, regression risks, or validation lanes.
---

# Model PR History Knowledge

This is a PR-driven knowledge base for model optimization history. It is not a
set of per-model skills. Each model family keeps bilingual docs with inspected
PR diffs, implementation file coverage, timelines, changed files, code excerpts,
and validation/risk notes.

Use it before patching model-specific serving paths, choosing an SGLang SOTA
optimization target, or explaining why a framework already has a faster path.

## Query

Run commands from this directory:

```bash
python3 scripts/query.py --list
python3 scripts/query.py --framework sglang --model qwen3-core --paths-only
python3 scripts/query.py --framework sglang --model qwen3-core "fused qk norm rope"
python3 scripts/query.py --framework vllm "DeepSeek-V4 fused norm router" --limit 5
python3 scripts/query.py --framework tokenspeed qwen35 --paths-only
```

Useful options:

- `--framework sglang|vllm|tensorrt_llm|tokenspeed`: restrict to one serving
  framework.
- `--model <slug>`: restrict to one model family directory.
- `--lang en|zh|both`: select English, Chinese, or both docs.
- `--paths-only`: print the exact docs to read without snippets.
- `--limit N`: bound search results.

## Workflow

1. Infer the model-family slug from the user's model id, checkpoint path, or
   SGLang source path. If unsure, run `scripts/query.py "<model name>"`.
2. Read the matching SGLang history first for SGLang patch work. Read competitor
   history too when vLLM, TensorRT-LLM, or TokenSpeed is the leading competitor
   or its trace suggests a missing SGLang fast path. If the doc opens with a
   dated `PR Backfill Audit` section, read it first: it lists the most recent
   PR-numbered merges that are not yet folded into the older timeline /
   diff-audit cards.
3. Extract only actionable evidence:
   - model implementation files and symbols
   - PRs that changed the hot source path
   - prior fusions, overlap work, quantization, MoE, attention, cache, sampler,
     or loader changes
   - open/watch PRs that may explain a known gap or pending support issue
   - validation lanes and regression risks implied by the PR cards
4. Save a short note in the active run artifacts, for example
   `history/model-pr-history-notes.md`, with paths read, PR numbers, source
   files, and the decision each item influenced.
5. Do not copy long PR cards into the final answer. Cite paths and summarize the
   relevant implementation/risk.

## Model Slugs

Current frameworks:

- `sglang`
- `vllm`
- `tensorrt_llm`
- `tokenspeed`

Current model-family slugs include:

```text
deepseek-ocr, deepseek-ocr-2, deepseek-v3-r1, deepseek-v31, deepseek-v32,
deepseek-v4, ernie45, gemma4, glm-vlm-ocr, glm45, glm46-glm47, glm5-glm51,
gpt-oss, intern-s1, internvl35, jina-reranker-m0, kimi, ling25, llada21,
llama31, llama33-70b, llama4, mimo-v2-flash, minimax, mistral-small-4,
mixtral-quark-int4fp8-moe, nemotron-super, qwen-vlm-omni-asr, qwen3-coder,
qwen3-core, qwen3-next, qwen35, ring25, step35
```

## SOTA Loop Contract

For `sglang-sota-humanize-loop`, this knowledge base is an early context
source:

- Read it after model identification and before patch planning.
- Include the history paths and key PR evidence in `analysis/root-cause.md` or
  `history/model-pr-history-notes.md`.
- If the profiler points at a known model path, check whether the history has
  prior changes on that file before writing a new patch.
- If a competitor is faster, search that competitor's model history for the
  same model family and stage before assuming the gap is kernel-local. Refresh
  live source/PRs for the exact target commit before patch planning when the
  comparison depends on latest upstream behavior.
