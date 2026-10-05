<div align="center">

# AI-Infra-Auto-Driven-SKILLS

**Agent-ready skills for LLM serving benchmarks, profiler triage, capacity
planning, model Day-0 support, code review, incident triage, and model PR
history across SGLang, vLLM, TensorRT-LLM, and TokenSpeed.**

[![GitHub stars](https://img.shields.io/github/stars/BBuf/AI-Infra-Auto-Driven-SKILLS?style=social)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/AI-Infra-Auto-Driven-SKILLS?style=flat-square)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/commits/main)
[![Skills](https://img.shields.io/badge/skills-10-2f80ed?style=flat-square)](#skills)
[![PR histories](https://img.shields.io/badge/pr_histories-118-2ea44f?style=flat-square)](#model-pr-history)
[![KDA-Pilot](https://img.shields.io/badge/sibling-KDA--Pilot-ff7b72?style=flat-square)](https://github.com/BBuf/KDA-Pilot)

</div>

Plain `SKILL.md` directories that give a coding agent the operational memory
for real AI-infra work: fair cross-framework benchmarks, kernel-level profiler
reads, operator FLOPs, model support plans, and the upstream PRs that already
solved a similar problem. Kernel campaigns live in the sibling
**[KDA-Pilot](https://github.com/BBuf/KDA-Pilot)**; per-model diffusion runs
live in [`sglang-diffusion-optimization-flows/`](sglang-diffusion-optimization-flows/).

## Skills

| Skill | Use it when |
| --- | --- |
| [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) | Find the best deployment command for one model across SGLang, vLLM, TensorRT-LLM, TokenSpeed under the same workload, GPUs, and SLA. |
| [`llm-serving-capacity-planner`](skills/llm-serving-capacity-planner/) | Explain startup memory, KV cache budget, request capacity, or OOM pressure from SGLang/vLLM logs. |
| [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) | Capture or read a torch profiler trace and get kernel, overlap-opportunity, and fusion-opportunity tables checked against a catalog of known SGLang/vLLM/TensorRT-LLM/TokenSpeed/FlashInfer optimizations. |
| [`llm-pipeline-analysis`](skills/llm-pipeline-analysis/) | Break a trace into forward passes, layers, and kernels with anchor boundaries and Perfetto ranges. |
| [`torch-profiler-layer-track`](skills/torch-profiler-layer-track/) | Add verified layer-number guides and compact GPU lanes to a trace for Perfetto navigation. |
| [`model-compute-simulation`](skills/model-compute-simulation/) | Estimate operator shapes, FLOPs, and MFU for a serving shape, or map kernels back to operators. |
| [`sglang-model-day0-support`](skills/model-optimization/sglang-model-day0-support/) | Turn a new model architecture into an SGLang Day-0 PR DAG, validation matrix, and release lock. |
| [`sglang-humanize-review`](skills/sglang-humanize-review/) | Review an SGLang PR the way maintainers do, grounded in the full human review corpus. |
| [`sglang-prod-incident-triage`](skills/sglang-prod-incident-triage/) | Turn queue growth, timeouts, wrong outputs, crashes, or stalls into a replay and the next debug step. |
| [`model-architecture-diagram`](skills/model-architecture-diagram/) | Return original public architecture diagrams for popular LLM, VLM, MoE, OCR, and diffusion families. |

## Model PR History

[`model-pr-optimization-history/`](model-pr-optimization-history/) is one
queryable knowledge base (installed as `model-pr-history-knowledge`) with
118 bilingual dossiers: SGLang 45, vLLM
44, TensorRT-LLM 15, TokenSpeed 14.
Each lists a model family's implementation files, every PR that changed them,
and per-PR evidence cards. New generated entries are explicitly marked as
source inventories pending manual diff review. Read it before
patching a model path or calling an optimization new.

```bash
cd model-pr-optimization-history
python3 scripts/query.py --list
python3 scripts/query.py --framework vllm "qwen3 fused qk norm"
```

Dossiers are regenerated from upstream git history with
`tools/rebuild_model_pr_history_from_git.py`; see
[`update_prompt.md`](update_prompt.md) for the full refresh procedure and
[`docs/upstream-source-contracts.md`](docs/upstream-source-contracts.md) for
the inspected source revisions.

## Install

Claude Code plugin:

```text
/plugin marketplace add BBuf/AI-Infra-Auto-Driven-SKILLS
/plugin install ai-infra-auto-driven-skills@ai-infra-auto-driven-skills
```

Any skill runtime (Claude Code, Codex, Kimi, ...): link or copy the skill
directories into its skill directory.

```bash
git clone https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git
cd AI-Infra-Auto-Driven-SKILLS
SKILL_DIR=~/.claude/skills   # or ${CODEX_HOME:-~/.codex}/skills
mkdir -p "$SKILL_DIR"
for d in skills/*/ skills/model-optimization/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$PWD/${d%/}" "$SKILL_DIR/$(basename "$d")"
done
ln -sfn "$PWD/model-pr-optimization-history" "$SKILL_DIR/model-pr-history-knowledge"
```

## Evidence Rules

- Benchmark rows record model, framework commit, GPUs, workload, rate or
  concurrency, SLA status, both commands, and raw artifacts.
- Profiler reports keep prefill and decode separate and never reuse an older
  trace for a new capture.
- Performance claims are scoped to the exact model, hardware, precision,
  workload, and framework revisions; accuracy is checked on the real path.
- Historical PR evidence keeps its audit date; a source refresh is not a GPU
  rerun.

## Related Projects

- **[KDA-Pilot](https://github.com/BBuf/KDA-Pilot)** hosts standalone kernel
  loops, kernel knowledge, and NCU workflows.

## Star History

<div align="center">

[![Star History Chart](https://star-history.dera.page/svg?repos=BBuf/AI-Infra-Auto-Driven-SKILLS&type=Date)](https://star-history.dera.page/#BBuf/AI-Infra-Auto-Driven-SKILLS&Date)

</div>
