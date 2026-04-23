# AI-Infra-Auto-Driven-SKILLS

Agent skills for SGLang/vLLM/TensorRT-LLM development, profiling, and production incident triage.

## Structure

```
skills/
├── model-optimization/
│   ├── sglang/
│   │   ├── sglang-model-pr-diff-dossier/
│   │   ├── sglang-deepseek-v3-r1-optimization/
│   │   ├── sglang-qwen3-core-optimization/
│   │   ├── sglang-glm45-optimization/
│   │   └── ...
│   └── vllm/
│       └── README.md
├── llm-serving-auto-benchmark/
│   ├── SKILL.md
│   ├── references/
│   │   ├── example-plan.yaml
│   │   ├── framework-matrix.md
│   │   ├── result-schema.md
│   │   └── version-notes.md
│   └── scripts/
│       └── compare_benchmark_results.py
├── h100/
│   └── SKILL.md
├── h100-sglang-diffusion/
│   └── SKILL.md
├── sglang-prod-incident-triage/
│   ├── SKILL.md
│   ├── references/
│   │   ├── decision-tree.md
│   │   ├── communication-hang-case-study.md
│   │   ├── endpoints-and-signals.md
│   │   ├── moe-shared-oob-case-study.md
│   │   ├── ttft-prefill-not-queue-case-study.md
│   │   └── replay-trace-profile.md
│   └── scripts/
│       ├── incident_artifact_tool.py
│       └── replay_trusted_request_dump.py
├── sglang-torch-profiler-analysis/
│   ├── SKILL.md
│   └── scripts/
│       ├── analyze_sglang_torch_profile.py
│       ├── analyze_sglang_llm_torch_profile.py
│       ├── analyze_sglang_profiler_overlap.py
│       └── profile_common.py
```

Model PR histories are framework-scoped:

```
model-pr-optimization-history/
├── sglang/
│   ├── audits/
│   │   └── qwen-glm-three-pass-audit-2026-04-23.md
│   ├── deepseek-v3-r1/
│   ├── qwen3-core/
│   ├── glm45/
│   └── ...
└── vllm/
    └── README.md
```

## Placeholders

The `h100` and `h100-sglang-diffusion` skills use placeholder values. Replace them before use:

| Placeholder         | Meaning                                                 |
| ------------------- | ------------------------------------------------------- |
| `<your-h100-host>`  | SSH host alias in `~/.ssh/config`                       |
| `<your-container>`  | Docker container name on the remote host                |
| `<your-repo-path>`  | Absolute path to the SGLang repo inside the container   |
| `<your-cache-path>` | Host path mounted as `/root/.cache`                     |
| `<your-hf-token>`   | Hugging Face access token (never commit the real value) |

## Model Optimization Skills

The model optimization handbook series is split first by serving framework, then by model family/generation. This keeps SGLang and future vLLM model dossiers from sharing one flat namespace.

SGLang model optimization skills live under `skills/model-optimization/sglang/`:

- `sglang-deepseek-v3-r1-optimization`
- `sglang-deepseek-v31-optimization`
- `sglang-deepseek-v32-optimization`
- `sglang-kimi-k2-k25-optimization`
- `sglang-minimax-m2-series-optimization`
- `sglang-qwen3-core-optimization`
- `sglang-qwen3-next-optimization`
- `sglang-qwen35-optimization`
- `sglang-qwen36-optimization`
- `sglang-qwen3-coder-optimization`
- `sglang-qwen-vlm-omni-asr-optimization`
- `sglang-qwen-image-optimization`
- `sglang-glm45-optimization`
- `sglang-glm46-glm47-optimization`
- `sglang-glm5-glm51-optimization`
- `sglang-glm-vlm-ocr-optimization`
- `sglang-model-pr-diff-dossier` records the mandatory production standard for model PR histories: read every PR diff and write motivation, implementation, code excerpt, and validation/risk.

Future vLLM model optimization skills should live under `skills/model-optimization/vllm/` and follow the same per-PR dossier standard.

## Model PR Optimization History

The `model-pr-optimization-history/` directory is framework-scoped.

SGLang bilingual model evolution notes live under `model-pr-optimization-history/sglang/`:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `kimi`
- `minimax`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `qwen3-coder`
- `qwen-vlm-omni-asr`
- `qwen-image`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `glm-vlm-ocr`
- `audits/qwen-glm-three-pass-audit-2026-04-23.md` records the 2026-04-23 SGLang docs, git/PR, and public-blog completeness sweep for the Qwen/GLM series.

Future vLLM model histories should live under `model-pr-optimization-history/vllm/`.

## Install

Copy the desired skill directory into your local skill path:

```bash
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
```
