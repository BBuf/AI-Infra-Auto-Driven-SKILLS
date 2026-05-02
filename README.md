<div align="center">

# AI-Infra-Auto-Driven-SKILLS

**Agent-ready playbooks for LLM serving benchmarks, torch-profiler triage,
SGLang optimization, production incidents, and model PR intelligence.**

[![GitHub stars](https://img.shields.io/github/stars/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/forks)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/AI-Infra-Auto-Driven-SKILLS?style=flat-square)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/commits/main)
[![Core skills](https://img.shields.io/badge/core_skills-7-2f80ed?style=flat-square)](#core-skills)
[![Model runbooks](https://img.shields.io/badge/model_runbooks-58-8250df?style=flat-square)](#model-optimization-catalog)
[![PR histories](https://img.shields.io/badge/pr_histories-58-2ea44f?style=flat-square)](#model-optimization-catalog)

</div>

This repository is built for AI infrastructure engineers who want agents to do
real work, not recite generic prompts.

It gives an agent the operational memory needed to benchmark SGLang, vLLM, and
TensorRT-LLM fairly; split prefill and decode profiler evidence; turn traces
into kernel and fusion opportunities; triage SGLang production incidents from a
replay; and keep model-family optimization history close to the code that
actually changed.

If this saves you one stale model-support assumption, one misleading profiler
trace, or one late-night benchmark loop, a star helps more AI-infra engineers
find it.

## Why Star It

| Signal | What makes it useful |
| --- | --- |
| **7 core operational skills** | Small, focused playbooks for benchmark search, profiler analysis, SOTA loops, incidents, architecture diagrams, and H100 runs. |
| **58 model optimization runbooks** | SGLang and vLLM model-family skills for DeepSeek, Qwen, GLM, Kimi, MiniMax, Llama, Mistral, Nemotron, and more. |
| **58 PR history dossiers** | Diff-backed model evolution notes that record what changed, where it changed, and what risks remain. |
| **Stage-separated profiler workflow** | Prefill and decode are profiled as separate workloads so hot kernels do not get misattributed. |
| **Framework-neutral benchmark schema** | Compare SGLang, vLLM, and TensorRT-LLM with the same workload, SLA, artifact layout, and result table. |
| **Profiler-to-action fusion catalog** | Connect torch-profiler rows to known SGLang/vLLM fusion, overlap, and torch.compile patterns. |
| **Replay-first incident triage** | Preserve evidence, reproduce the request path, and choose the next debug tool before patching. |

## What You Can Do

| Goal | Start here |
| --- | --- |
| Search the best serving command across frameworks | [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) |
| Explain a torch-profiler trace with kernel, overlap, and fusion tables | [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) |
| Drive a full SGLang performance loop against vLLM/TensorRT-LLM | [`sglang-sota-performance`](skills/sglang-sota-performance/) |
| Debug a live or recent SGLang serving incident from evidence | [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) |
| Find original public model architecture diagrams | [`model-architecture-diagram`](skills/model-architecture-diagram/) |
| Reuse model-family optimization knowledge | [`skills/model-optimization`](skills/model-optimization/) |
| Read model PR evolution by framework | [`model-pr-optimization-history`](model-pr-optimization-history/) |

## Core Skills

| Skill | Use it when |
| --- | --- |
| [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) | You need a fair, bounded serving benchmark search for SGLang, vLLM, TensorRT-LLM, or another OpenAI-compatible stack. |
| [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) | You need a three-table profiler report that keeps `extend/prefill` and `decode` evidence separate. |
| [`sglang-sota-performance`](skills/sglang-sota-performance/) | You want SGLang to match or beat the best observed framework result for a specific model and workload. |
| [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) | You need to turn queue growth, timeouts, wrong outputs, crashes, or distributed stalls into a replay and next debug step. |
| [`model-architecture-diagram`](skills/model-architecture-diagram/) | You need original public architecture diagrams for popular LLM, VLM, MoE, OCR, and diffusion model families. |
| [`h100`](skills/h100/) | You need an H100 operator runbook for SGLang validation in the configured remote environment. |
| [`h100-sglang-diffusion`](skills/h100-sglang-diffusion/) | You need the H100 workflow with diffusion-specific paths and validation expectations. |

## Model Optimization Catalog

The model optimization layer is intentionally larger than the core skill set.
Core skills teach an agent how to work; model runbooks teach it what has
already happened for each model family.

| Framework | Runbooks | PR histories |
| --- | ---: | ---: |
| [SGLang](skills/model-optimization/sglang/) | 29 | 29 |
| [vLLM](skills/model-optimization/vllm/) | 29 | 29 |

Covered families include:

```text
DeepSeek V3/R1/V3.1/V3.2/V4, Qwen3, Qwen3-Coder, Qwen3-Next,
Qwen3.5/Qwen3.6, Qwen VLM/Omni/ASR, GLM 4.5/4.6/4.7/5,
Kimi, MiniMax, Llama 4, Mistral Small 4, Mixtral, Nemotron,
Gemma, Ernie 4.5, Intern-S1, InternVL, Hunyuan, MOSS-VL,
GPT-OSS, Step 3.5, Mimo, and model-specific MoE/quantization paths.
```

Each model-family history is designed to answer practical questions:

- Which PRs changed this model path?
- Was the PR merged, closed, or still open?
- Which files and symbols moved?
- What optimization or correctness risk should be checked before touching it?
- Which upstream idea should be compared before writing a new kernel or fusion?

## Evidence Standards

The repo is opinionated about evidence because performance work gets noisy fast.

- Benchmark rows should include model, framework, GPU count, workload, request
  rate or concurrency, SLA status, launch command, benchmark command, and raw
  artifacts.
- Profiler reports should keep prefill and decode separate, then emit the same
  three tables: kernel table, overlap-opportunity table, and fuse-opportunity
  table.
- SOTA claims should be scoped to the exact model, hardware, framework commits,
  precision, workload, and SLA used in the run.
- Incident triage should start from replayable evidence instead of changing code
  from symptoms alone.
- Model optimization notes should point back to PRs, files, diffs, and risk
  surfaces rather than vague summary text.

## Install

Copy only the skills you want into your agent skill directory:

```bash
cp -r skills/llm-serving-auto-benchmark <agent-skill-dir>/llm-serving-auto-benchmark
cp -r skills/llm-torch-profiler-analysis <agent-skill-dir>/llm-torch-profiler-analysis
cp -r skills/sglang-sota-performance <agent-skill-dir>/sglang-sota-performance
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/model-architecture-diagram <agent-skill-dir>/model-architecture-diagram
```

Install a model-family skill when you are working on that exact family:

```bash
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
cp -r skills/model-optimization/vllm/vllm-qwen3-core-optimization <agent-skill-dir>/vllm-qwen3-core-optimization
```

The H100 skills document a concrete operator environment. If you adapt them,
replace the SSH alias, container name, and workspace paths in one pass, and keep
secrets such as Hugging Face tokens out of the repository.

## Repository Map

```text
skills/
├── llm-serving-auto-benchmark/      # serving benchmark search and comparison
├── llm-torch-profiler-analysis/     # profiler capture and trace triage
├── sglang-sota-performance/         # end-to-end SGLang optimization loop
├── sglang-prod-incident-triage/     # replay-first serving incident workflow
├── model-architecture-diagram/      # public architecture diagram resolver
├── h100/                            # H100 operator runbook
├── h100-sglang-diffusion/           # H100 diffusion operator runbook
└── model-optimization/
    ├── model-pr-diff-dossier/       # shared PR dossier standard
    ├── sglang/                      # 29 SGLang model-family runbooks
    └── vllm/                        # 29 vLLM model-family runbooks

model-pr-optimization-history/
├── sglang/                          # 29 SGLang model-family histories
└── vllm/                            # 29 vLLM model-family histories
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BBuf/AI-Infra-Auto-Driven-SKILLS&type=Date)](https://star-history.com/#BBuf/AI-Infra-Auto-Driven-SKILLS&Date)
