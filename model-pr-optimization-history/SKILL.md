---
name: model-pr-history-knowledge
description: Use when an SGLang, vLLM, TensorRT-LLM, or TokenSpeed serving/model optimization task needs prior model-family PR evidence. Query and read the PR-driven history docs under model-pr-optimization-history before choosing source paths, fast paths, kernel/fusion ideas, regression risks, or validation lanes.
---

# Model PR History Knowledge

This is a PR-driven knowledge base for model optimization history. It is not a
set of per-model skills. Each model family keeps bilingual docs with inspected
PR diffs, implementation file coverage, timelines, changed files, code excerpts,
and validation/risk notes.

Use it before patching model-specific serving paths, choosing an optimization
target, or explaining why a framework already has a faster path.

Use the maintained [source contracts](../docs/upstream-source-contracts.md)
when applying historical evidence to a current branch. The review/history
corpora retain their own capture dates; they do not certify today's dispatch
or numerical defaults. For kernel replacements, verify that the real model
executes the candidate before treating end-to-end tests as coverage.

Generated entries labeled **not a manual audit** are PR discovery inventories.
Read their full diffs and current callers before making optimization conclusions;
API file counts and snippets do not establish complete diff coverage. Existing
manual audit notes retain their original dates.

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
   dated note (source-head refresh, backfill audit, or reviewed kernel
   addendum), read it first: it carries hand-reviewed context that the
   generated timeline and cards below do not.
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

- `sglang` (45 families)
- `vllm` (44 families)
- `tensorrt_llm` (15 families)
- `tokenspeed` (14 families)

Current model-family slugs include:

```text
deepseek-ocr, deepseek-ocr-2, deepseek-v3-r1, deepseek-v31, deepseek-v32,
deepseek-v4, deepseek-v41, dots3, ernie45, exaone4, gemma4, glm-vlm-ocr,
glm45, glm46-glm47, glm5-glm51, gpt-oss, hunyuan3-preview, hunyuan4, inkling,
intern-s1, internvl35, jina-reranker-m0, kimi, ling25, ling3, llada21,
llama31, llama33-70b, llama4, longcat-flash, mimo-v2-flash, minimax,
mistral-small-4, mixtral-quark-int4fp8-moe, moss-vl, nemotron-super,
qwen-vlm-omni-asr, qwen3-coder, qwen3-core, qwen3-next, qwen35, qwen36,
qwen38, qwen4-exp, ring25, step35, step37
```

Availability is framework-specific; `python3 scripts/query.py --list` is the
authority. Notes on overlapping slugs:

- `deepseek-v4` keeps every V4 PR; `deepseek-v41` is the V4.1 subset of the
  same files (subject must name V4.1). Read both for DSV4.1 work.
- `glm5-glm51` covers GLM-5, 5.1, 5.2 and 5.3-Flash (`glm5_next`,
  `glm53_flash`, `GlmMoeDsa`).
- `qwen4-exp` is the `qwen4_exp` model file used by Qwen3.8-Flash-Next;
  `qwen38` tracks the Qwen3.8 cookbook surface. Public `Qwen/Qwen3.8-27B` is
  `model_type=qwen3_5`, so its loader and kernel history is under `qwen35`
  (vLLM has no dedicated `qwen38` slug).
- `hunyuan4` is Hy4 (`hunyuan_v4` / `hy_v4`); `ling3` is BailingMoeV3;
  `nemotron-super` also covers Nemotron-H; `minimax` covers M2 and M3; `kimi`
  covers K2, K2.5, K3, Linear and VL.
- TensorRT-LLM and TokenSpeed dossiers are generated the same way as SGLang and
  vLLM. Their dated hand-written refresh notes stay at the top of the page.

## Optimization Workflow Contract

When a profiling or optimization task uses this knowledge base:

- Read it after model identification and before patch planning.
- Record the history paths and key PR evidence in the run notes (for example
  `history/model-pr-history-notes.md`).
- If the profiler points at a known model path, check whether the history has
  prior changes on that file before writing a new patch.
- If a competitor is faster, search that competitor's model history for the
  same model family and stage before assuming the gap is kernel-local. Refresh
  live source/PRs for the exact target commit when the comparison depends on
  the latest upstream behavior.
