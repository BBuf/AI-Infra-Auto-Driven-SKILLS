<div align="center">

# AI-Infra-Auto-Driven-SKILLS

**Agent-ready playbooks for LLM serving benchmarks, torch-profiler triage,
SGLang optimization, production incidents, and model PR intelligence.**

[![GitHub stars](https://img.shields.io/github/stars/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/forks)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/AI-Infra-Auto-Driven-SKILLS?style=flat-square)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/commits/main)
[![Core skills](https://img.shields.io/badge/core_skills-6-2f80ed?style=flat-square)](#core-skills)
[![PR histories](https://img.shields.io/badge/pr_histories-58-2ea44f?style=flat-square)](#model-pr-history-catalog)
[![KernelPilot](https://img.shields.io/badge/sibling-KernelPilot-ff7b72?style=flat-square)](https://github.com/BBuf/kernel-pilot)

</div>

This repository is built for AI infrastructure engineers who want agents to do
real work, not recite generic prompts.

It gives an agent the operational memory needed to benchmark SGLang, vLLM, and
TensorRT-LLM fairly; split prefill and decode profiler evidence; turn traces
into kernel and fusion opportunities; triage SGLang production incidents from a
replay; and keep model-family optimization history close to the code that
actually changed.

For standalone kernel campaigns and kernel evidence tools, see the sibling
project **[KernelPilot](https://github.com/BBuf/kernel-pilot)**.

If this saves you one stale model-support assumption, one misleading profiler
trace, or one late-night benchmark loop, a star helps more AI-infra engineers
find it.

## Why Star It

| Signal | What makes it useful |
| --- | --- |
| **6 core operational skills** | Small, focused playbooks for benchmark search, profiler analysis, Humanize-governed SOTA loops, incidents, architecture diagrams, and PR history. |
| **58 PR history dossiers** | A queryable, PR-driven model history knowledge base that records what changed, where it changed, and what risks remain. |
| **Stage-separated profiler workflow** | Prefill and decode are profiled as separate workloads so hot kernels do not get misattributed. |
| **Framework-neutral benchmark schema** | Compare SGLang, vLLM, and TensorRT-LLM with the same workload, SLA, artifact layout, and result table. |
| **Profiler-to-action fusion catalog** | Connect torch-profiler rows to known SGLang/vLLM fusion, overlap, and torch.compile patterns. |
| **Replay-first incident triage** | Preserve evidence, reproduce the request path, and choose the next debug tool before patching. |
| **KernelPilot sibling project** | Link out to [KernelPilot](https://github.com/BBuf/kernel-pilot) for standalone kernel loops, kernel knowledge, and NCU report workflows. |

## What You Can Do

| Goal | Start here |
| --- | --- |
| Search the best serving command across frameworks | [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) |
| Explain a torch-profiler trace with kernel, overlap, and fusion tables | [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) |
| Drive a full SGLang performance loop against vLLM/TensorRT-LLM | [`sglang-sota-performance`](skills/sglang-sota-performance/) |
| Turn the SGLang SOTA loop into one Humanize-governed model patch loop | [`sglang-sota-humanize-loop`](skills/sglang-sota-humanize-loop/) |
| Debug a live or recent SGLang serving incident from evidence | [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) |
| Run standalone kernel optimization loops or query kernel evidence | [`KernelPilot`](https://github.com/BBuf/kernel-pilot) |
| Find original public model architecture diagrams | [`model-architecture-diagram`](skills/model-architecture-diagram/) |
| Query PR-driven model optimization history by framework | [`model-pr-optimization-history`](model-pr-optimization-history/) |

## Core Skills

| Skill | Use it when |
| --- | --- |
| [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) | You need a fair, bounded serving benchmark search for SGLang, vLLM, TensorRT-LLM, or another OpenAI-compatible stack. |
| [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) | You need a three-table profiler report that keeps `extend/prefill` and `decode` evidence separate. |
| [`sglang-sota-performance`](skills/sglang-sota-performance/) | You want SGLang to match or beat the best observed framework result for a specific model and workload. |
| [`sglang-sota-humanize-loop`](skills/sglang-sota-humanize-loop/) | You want the SGLang SOTA workflow to run as one model-level Humanize RLCR loop after the fixed fair benchmark and profiler gate, with KernelPilot knowledge and `ncu-report` as kernel assists only. |
| [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) | You need to turn queue growth, timeouts, wrong outputs, crashes, or distributed stalls into a replay and next debug step. |
| [`model-architecture-diagram`](skills/model-architecture-diagram/) | You need original public architecture diagrams for popular LLM, VLM, MoE, OCR, and diffusion model families. |

## Model PR History Catalog

The model optimization layer is now one knowledge base:
[`model-pr-optimization-history`](model-pr-optimization-history/). It contains
58 PR-driven history dossiers and a small query helper. These are not
per-model runbook skills; they preserve diff-backed model evolution records for
SGLang and vLLM so SOTA loops can read prior source and PR evidence before
patching.

| Framework | PR histories |
| --- | ---: |
| [SGLang](model-pr-optimization-history/sglang/) | 29 |
| [vLLM](model-pr-optimization-history/vllm/) | 29 |

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

Query examples:

```bash
cd model-pr-optimization-history
python3 scripts/query.py --list
python3 scripts/query.py --framework sglang --model qwen3-core --paths-only
python3 scripts/query.py --framework vllm "qwen3 fused qk norm"
```

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
- Humanize SGLang SOTA loops should keep the fair benchmark and required
  profiler evidence outside the patch loop, then use one model-level RLCR loop
  for SGLang code changes.
- Kernel-local SGLang fixes inside that loop should cite KernelPilot knowledge
  pages or PR bundles when they influence code, store NCU digests when counter
  evidence is needed, and still pass the same real-model benchmark/profile gate.
- Incident triage should start from replayable evidence instead of changing code
  from symptoms alone.
- Model optimization histories should point back to PRs, files, diffs, and risk
  surfaces rather than vague summary text; they live as one PR-driven knowledge
  base, not per-model skills.

## Install

Copy only the skills you want into your agent skill directory:

```bash
cp -r skills/llm-serving-auto-benchmark <agent-skill-dir>/llm-serving-auto-benchmark
cp -r skills/llm-torch-profiler-analysis <agent-skill-dir>/llm-torch-profiler-analysis
cp -r skills/sglang-sota-performance <agent-skill-dir>/sglang-sota-performance
cp -r skills/sglang-sota-humanize-loop <agent-skill-dir>/sglang-sota-humanize-loop
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/model-architecture-diagram <agent-skill-dir>/model-architecture-diagram
cp -r model-pr-optimization-history <agent-skill-dir>/model-pr-history-knowledge
```

Copy `model-pr-optimization-history` only when you want the agent to query the
PR-driven model knowledge base locally. It replaces the old per-model runbook
skill layout with one shared knowledge root.

## Repository Map

```text
skills/
├── llm-serving-auto-benchmark/      # serving benchmark search and comparison
├── llm-torch-profiler-analysis/     # profiler capture and trace triage
├── sglang-sota-performance/         # end-to-end SGLang optimization loop
├── sglang-sota-humanize-loop/       # Humanize-governed SGLang SOTA loop
├── sglang-prod-incident-triage/     # replay-first serving incident workflow
├── model-architecture-diagram/      # public architecture diagram resolver
└── model-optimization/
    └── model-pr-diff-dossier/       # shared PR history quality standard

model-pr-optimization-history/
├── SKILL.md                         # knowledge-base usage instructions
├── scripts/query.py                 # local model/keyword query helper
├── sglang/                          # 29 PR-driven SGLang model histories
└── vllm/                            # 29 PR-driven vLLM model histories
```

## Related Projects

- **[Humanize](https://github.com/PolyArch/humanize)** provides the RLCR
  workflow that powers the Humanize-governed SGLang SOTA loop.
- **[KernelPilot](https://github.com/BBuf/kernel-pilot)** is the sibling home
  for standalone kernel loops, kernel knowledge, and NCU report workflows.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BBuf/AI-Infra-Auto-Driven-SKILLS&type=Date)](https://star-history.com/#BBuf/AI-Infra-Auto-Driven-SKILLS&Date)
