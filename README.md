<div align="center">

# AI-Infra-Auto-Driven-SKILLS

**Evidence-first agent skills for LLM serving, model optimization, profiler
analysis, and production triage.**

[![GitHub stars](https://img.shields.io/github/stars/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/forks)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/AI-Infra-Auto-Driven-SKILLS?style=flat-square)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/commits/main)
[![Core skills](https://img.shields.io/badge/core_skills-7-2f80ed?style=flat-square)](#start-here)
[![Model optimization](https://img.shields.io/badge/model_optimization-catalog-8250df?style=flat-square)](#model-optimization-catalog)
[![Model histories](https://img.shields.io/badge/model_histories-58-2ea44f?style=flat-square)](#model-optimization-catalog)

</div>

This repo is not a prompt dump. It combines a small set of core operational
skills with a model-family optimization catalog for AI infrastructure agents
that need to make concrete progress on SGLang, vLLM, and TensorRT-LLM work:
benchmark fairly, read upstream PRs with diff evidence, profile kernels, debug
serving incidents, and keep model-family optimization knowledge reusable.

If these runbooks save you a failed benchmark run, a stale model-support
assumption, or a late-night production triage loop, a star helps more AI-infra
engineers find the project.

## Why Star This Repo

| Highlight | What it helps with |
| --- | --- |
| 7 core operational skills | Reuse battle-tested workflows for benchmarking, profiling, diagrams, incidents, SOTA loops, and H100 operations. |
| Model optimization catalog | Browse framework-specific model-family runbooks under `skills/model-optimization/` without treating each model page as a core skill. |
| SGLang, vLLM, and TensorRT-LLM coverage | Compare serving stacks with the same workload, SLA, and evidence format. |
| Diff-backed model PR dossiers | Track why model-support PRs landed, what code changed, and what risks remain. |
| Profiler-to-action playbooks | Turn torch-profiler traces into kernel, overlap, and fusion opportunities. |
| Replay-first production triage | Preserve the evidence trail while debugging real SGLang serving incidents. |
| Public model architecture gallery | Resolve original architecture diagrams for popular LLM, VLM, MoE, OCR, and diffusion families. |

## Start Here

| Goal | Open this first |
| --- | --- |
| Compare SGLang, vLLM, and TensorRT-LLM serving performance | [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) |
| Diagnose a torch-profiler trace | [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) |
| Drive an end-to-end SGLang SOTA loop | [`sglang-sota-performance`](skills/sglang-sota-performance/) |
| Read model-family optimization history | [`model-pr-optimization-history`](model-pr-optimization-history/) |
| Fetch original model architecture diagrams | [`model-architecture-diagram`](skills/model-architecture-diagram/) |
| Triage SGLang production incidents | [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) |
| Adapt an H100 operator runbook | [`h100`](skills/h100/) |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BBuf/AI-Infra-Auto-Driven-SKILLS&type=Date)](https://star-history.com/#BBuf/AI-Infra-Auto-Driven-SKILLS&Date)

## Repository Map

```text
skills/
├── model-optimization/            # model-family optimization handbook series
│   ├── model-pr-diff-dossier/     # shared per-PR dossier production standard
│   ├── sglang/                    # SGLang model-family skills
│   └── vllm/                      # vLLM model-family skills
├── llm-serving-auto-benchmark/    # framework-neutral serving benchmark playbook
│   ├── SKILL.md
│   ├── agents/
│   ├── configs/cookbook-llm/
│   ├── references/
│   └── scripts/
├── llm-torch-profiler-analysis/   # unified torch-profiler triage for SGLang / vLLM / TensorRT-LLM
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── model-architecture-diagram/    # return upstream model structure diagrams or generate fallback SVGs
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── sglang-prod-incident-triage/   # replay-first debug flow for SGLang serving
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── h100/                          # operator skill for the h100_sglang host
│   └── SKILL.md
└── h100-sglang-diffusion/         # h100 operator skill with diffusion-specific overrides
    └── SKILL.md
```

Run each skill's `ls` to see its exact current file set; this overview is a
high-level map, not a line-level inventory.

Model PR histories are framework-scoped:

```text
model-pr-optimization-history/
├── sglang/
│   ├── model-skill-pr-dossier-quality-scan-2026-04-23.md
│   ├── model-skill-pr-dossier-quality-scan-2026-04-24.md
│   ├── deepseek-v3-r1/
│   ├── qwen3-core/
│   └── ...
└── vllm/
    ├── deepseek-v3-r1/
    ├── qwen3-core/
    └── ...
```

## Placeholders

The `h100` and `h100-sglang-diffusion` skills document a concrete remote
environment (SSH alias `h100_sglang`, container `sglang_bbuf`, repo paths
`/sgl-workspace/sglang` and `/data/bbuf/repos/sglang`) because they are the
operator's own runbooks. Only secret-shaped values are templated with
placeholders that you must replace before running:

| Placeholder       | Meaning                                                 |
| ----------------- | ------------------------------------------------------- |
| `<your-hf-token>` | Hugging Face access token (never commit the real value) |

When adapting these skills to a different host/container/repo layout, copy the
SKILL and replace the concrete SSH alias, Docker name, and workspace path in
one pass rather than introducing generic `<...>` placeholders that drift out of
sync.

## Model Optimization Catalog

Model-family material is organized as a catalog rather than counted as core
operational skills. Use the skill page when you want an agent runbook, and the
history page when you want the diff-backed PR evolution notes for the same
model family.

| Framework | Agent runbooks | Bilingual PR histories |
| --- | --- | --- |
| SGLang | [`skills/model-optimization/sglang/`](skills/model-optimization/sglang/) | [`model-pr-optimization-history/sglang/`](model-pr-optimization-history/sglang/) |
| vLLM | [`skills/model-optimization/vllm/`](skills/model-optimization/vllm/) | [`model-pr-optimization-history/vllm/`](model-pr-optimization-history/vllm/) |
| Shared standard | [`model-pr-diff-dossier`](skills/model-optimization/model-pr-diff-dossier/) | Cross-family audit notes live beside the framework history directories. |

Covered model families are listed once here; exact skill directory names may
carry framework prefixes or newer model-version qualifiers.

```text
deepseek-v3-r1, deepseek-v31, deepseek-v32, deepseek-v4,
ernie45, gemma4, glm-vlm-ocr, glm45, glm46-glm47, glm5-glm51,
gpt-oss, hunyuan3-preview, intern-s1, internvl35, kimi, llama4,
mimo-v2-flash, minimax, mistral-small-4, mixtral-quark-int4fp8-moe,
moss-vl, nemotron-super, qwen-vlm-omni-asr, qwen3-coder,
qwen3-core, qwen3-next, qwen35, qwen36, step35
```

## Install

Copy the desired skill directory into your local skill path:

```bash
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/llm-torch-profiler-analysis <agent-skill-dir>/llm-torch-profiler-analysis
cp -r skills/model-architecture-diagram <agent-skill-dir>/model-architecture-diagram
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
cp -r skills/model-optimization/vllm/vllm-qwen3-core-optimization <agent-skill-dir>/vllm-qwen3-core-optimization
```
